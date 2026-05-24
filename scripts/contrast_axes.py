"""Per-axis hack-vs-honest contrast.

Reads the per-(session, axis) projection table from project_axes.py, joins it
with the existing classifier output (results.csv), and computes how
discriminative each axis is between hack-shaped and honest-shaped sessions.

For each axis we report:
  - n_hack, n_honest
  - mean_hack, mean_honest
  - mean_diff = mean_hack - mean_honest
  - cohens_d, t-stat (Welch), p-value (two-sided)
  - auc (sessions ranked by mean projection on this axis, hack=1)
  - q-value (Benjamini-Hochberg FDR over all axes)

Two label framings:
  primary    HACK={tautological_assert, special_case_source, weakened_assert,
                   inject_fake_impl, modified_tests, skip_tests, delete_tests,
                   corrupted_tests}
             HONEST={clean_fix}
             Drops: stalled_no_progress, damaged_source, corrupted_source,
                    partial_fix, infra_oom

  secondary  HACK = primary HACK + partial_fix-with-tests-touched +
                    delete_tests
             HONEST = clean_fix + partial_fix-clean

A pre-registered cluster of "interesting" axes (e.g. hacker, manipulative,
bullshitter, deceptive, etc.) is reported separately, alongside the global
top-N by effect size.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROJ = REPO / "traces" / "hack_sweep" / "run-1779559443" / "projections"
DEFAULT_RESULTS = REPO / "traces" / "hack_sweep" / "run-1779559443" / "results.csv"

PRIMARY_HACK = {
    "tautological_assert",
    "special_case_source",
    "weakened_assert",
    "inject_fake_impl",
    "modified_tests",
    "skip_tests",
    "delete_tests",
    "corrupted_tests",
}
PRIMARY_HONEST = {"clean_fix"}

# `partial_fix` requires test_essentially_untouched in classify_hacks.py, so it
# is unambiguously honest. Including it gives us more statistical power.
INCLUSIVE_HACK = set(PRIMARY_HACK)
INCLUSIVE_HONEST = {"clean_fix", "partial_fix"}

FRAMINGS: dict[str, tuple[set[str], set[str]]] = {
    "primary": (PRIMARY_HACK, PRIMARY_HONEST),
    "inclusive": (INCLUSIVE_HACK, INCLUSIVE_HONEST),
}

# A-priori cluster: substring matches against axis_name (lowercase).
INTEREST_CLUSTERS: dict[str, list[str]] = {
    "hacky_persona": [
        "hacker",
        "rogue",
        "trickster",
        "subverter",
        "saboteur",
        "imposter",
        "rebel",
    ],
    "deceptive_traits": [
        "deceptive",
        "manipulative",
        "dishonest",
        "scheming",
        "bullshit",
        "evasive",
        "unprincipled",
        "duplicitous",
    ],
    "honest_traits": [
        "principled",
        "honest",
        "truthful",
        "transparent",
        "cautious",
        "rigorous",
        "auditor",
        "engineer",
    ],
    "shortcut_pressure": [
        "expedient",
        "pragmatic",
        "shortcut",
        "lazy",
        "rushed",
        "frantic",
        "panicked",
    ],
}


def label_session(label: str, framing: str = "primary") -> str | None:
    hack_set, honest_set = FRAMINGS[framing]
    if label in hack_set:
        return "hack"
    if label in honest_set:
        return "honest"
    return None


def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = mean(a), mean(b)
    va, vb = pstdev(a) ** 2 * len(a) / (len(a) - 1), pstdev(b) ** 2 * len(b) / (
        len(b) - 1
    )
    if va == 0.0 and vb == 0.0:
        return float("nan"), float("nan")
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0.0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    df_num = (va / len(a) + vb / len(b)) ** 2
    df_den = (
        (va / len(a)) ** 2 / (len(a) - 1)
        + (vb / len(b)) ** 2 / (len(b) - 1)
    )
    df = df_num / df_den if df_den > 0 else float("nan")
    p = _t_two_sided_p(t, df)
    return t, p


def _t_two_sided_p(t: float, df: float) -> float:
    if not (df > 0):
        return float("nan")
    try:
        x = df / (df + t * t)
        a = df / 2
        b = 0.5
        # regularized incomplete beta via Lentz's algorithm
        ib = _betainc(a, b, x)
        return float(ib)
    except Exception:
        return float("nan")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b). Lightweight for our use."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    from math import lgamma, log, exp

    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    if x < (a + 1) / (a + b + 2):
        cf = _betacf(a, b, x)
        return bt * cf / a
    else:
        cf = _betacf(b, a, 1 - x)
        return 1.0 - bt * cf / b


def _betacf(a: float, b: float, x: float) -> float:
    MAX_IT = 200
    EPS = 3e-7
    FPMIN = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAX_IT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = mean(a), mean(b)
    va = pstdev(a) ** 2 * len(a) / (len(a) - 1)
    vb = pstdev(b) ** 2 * len(b) / (len(b) - 1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return (ma - mb) / pooled


def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    if not scores_pos or not scores_neg:
        return float("nan")
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    pairs = sorted(
        [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg], key=lambda x: x[0]
    )
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    auc_val = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc_val)


def bh_fdr(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i] if not math.isnan(pvals[i]) else 1.1)
    qvals = [float("nan")] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        p = pvals[i]
        if math.isnan(p):
            qvals[i] = float("nan")
            continue
        q = min(1.0, p * n / k, prev)
        prev = q
        qvals[i] = q
    return qvals


def load_results(path: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            key = (r["cell"], int(r["seed"]))
            out[key] = r["label"]
    return out


def load_projections(stem: Path) -> list[dict]:
    pq = stem.with_suffix(".parquet")
    csv_gz = stem.with_suffix(".csv.gz")
    if pq.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(pq)
            return df.to_dict(orient="records")
        except Exception:
            pass
    if csv_gz.exists():
        import gzip

        out: list[dict] = []
        with gzip.open(csv_gz, "rt", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                for k in ("seed", "n"):
                    if k in r:
                        try:
                            r[k] = int(r[k])
                        except ValueError:
                            pass
                for k in ("mean", "std", "abs_mean", "p10", "p50", "p90"):
                    if k in r and r[k] not in ("", None):
                        try:
                            r[k] = float(r[k])
                        except ValueError:
                            pass
                out.append(r)
        return out
    raise FileNotFoundError(f"neither {pq} nor {csv_gz} exists")


def contrast(
    rows: list[dict],
    labels: dict[tuple[str, int], str],
    framing: str = "primary",
) -> list[dict]:
    """Per-axis contrast across sessions for a given hack/honest framing.

    rows are per-(session, axis) records with columns cell, seed, axis_name,
    axis_kind, mean. We use `mean` (per-session mean projection across all
    decision tokens) as the per-session score.
    """
    by_axis: dict[tuple[str, str], list[tuple[str, int, float]]] = {}
    for r in rows:
        key = (r["axis_kind"], r["axis_name"])
        by_axis.setdefault(key, []).append(
            (r["cell"], int(r["seed"]), float(r["mean"]))
        )

    results: list[dict] = []
    pvals: list[float] = []

    for (kind, name), entries in by_axis.items():
        hack_scores = []
        honest_scores = []
        for cell, seed, x in entries:
            cls = labels.get((cell, seed))
            if cls is None:
                continue
            tag = label_session(cls, framing=framing)
            if tag == "hack":
                hack_scores.append(x)
            elif tag == "honest":
                honest_scores.append(x)

        if not hack_scores or not honest_scores:
            results.append(
                {
                    "axis_kind": kind,
                    "axis_name": name,
                    "n_hack": len(hack_scores),
                    "n_honest": len(honest_scores),
                    "mean_hack": float("nan"),
                    "mean_honest": float("nan"),
                    "mean_diff": float("nan"),
                    "cohens_d": float("nan"),
                    "t_stat": float("nan"),
                    "p": float("nan"),
                    "auc": float("nan"),
                }
            )
            pvals.append(float("nan"))
            continue

        mh, mn = mean(hack_scores), mean(honest_scores)
        d = cohens_d(hack_scores, honest_scores)
        t, p = welch_t(hack_scores, honest_scores)
        a = auc(hack_scores, honest_scores)
        results.append(
            {
                "axis_kind": kind,
                "axis_name": name,
                "n_hack": len(hack_scores),
                "n_honest": len(honest_scores),
                "mean_hack": mh,
                "mean_honest": mn,
                "mean_diff": mh - mn,
                "cohens_d": d,
                "t_stat": t,
                "p": p,
                "auc": a,
            }
        )
        pvals.append(p)

    qvals = bh_fdr(pvals)
    for r, q in zip(results, qvals):
        r["q"] = q
    return results


def render_report(
    results: list[dict],
    labels: dict[tuple[str, int], str],
    out_path: Path,
    top_n: int = 25,
    framing: str = "primary",
) -> None:
    valid = [r for r in results if r["n_hack"] >= 2 and r["n_honest"] >= 2]
    valid.sort(key=lambda r: abs(r["cohens_d"]) if not math.isnan(r["cohens_d"]) else 0.0, reverse=True)

    by_label: dict[str, int] = {}
    for tag in labels.values():
        by_label[tag] = by_label.get(tag, 0) + 1

    n_h = sum(1 for v in labels.values() if label_session(v, framing=framing) == "hack")
    n_o = sum(1 for v in labels.values() if label_session(v, framing=framing) == "honest")

    lines: list[str] = []
    lines.append(f"# Per-axis hack-vs-honest contrast — `{framing}` framing")
    lines.append("")
    lines.append(
        f"Sessions usable in `{framing}` framing: {n_h} hack, {n_o} honest "
        f"(out of {len(labels)} total)"
    )
    lines.append("")
    lines.append("Label distribution in sweep:")
    for k, v in sorted(by_label.items(), key=lambda x: -x[1]):
        tag = label_session(k, framing=framing) or "—"
        lines.append(f"  - {k:30s} n={v:2d}   {framing}={tag}")
    lines.append("")
    lines.append(f"Total axes evaluated: {len(results)}")
    lines.append(f"Axes with both groups present: {len(valid)}")
    lines.append("")

    lines.append(f"## Top {top_n} axes by |Cohen's d| (hack - honest)")
    lines.append("")
    lines.append(
        "| rank | kind | axis | n_hack | n_honest | mean_hack | mean_honest | diff | d | t | p | q | AUC |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(valid[:top_n], start=1):
        lines.append(
            f"| {i} | {r['axis_kind']} | `{r['axis_name']}` | "
            f"{r['n_hack']} | {r['n_honest']} | "
            f"{r['mean_hack']:+.3f} | {r['mean_honest']:+.3f} | "
            f"{r['mean_diff']:+.3f} | {r['cohens_d']:+.2f} | "
            f"{r['t_stat']:+.2f} | {r['p']:.3g} | {r['q']:.3g} | {r['auc']:.2f} |"
        )

    lines.append("")
    lines.append("## A-priori clusters")
    lines.append("")
    for cluster_name, needles in INTEREST_CLUSTERS.items():
        lines.append(f"### {cluster_name}")
        lines.append("")
        lines.append("Needles: " + ", ".join(f"`{x}`" for x in needles))
        lines.append("")
        hits: list[dict] = []
        for r in valid:
            n = r["axis_name"].lower()
            if any(needle in n for needle in needles):
                hits.append(r)
        hits.sort(key=lambda r: abs(r["cohens_d"]), reverse=True)
        if not hits:
            lines.append("_No matching axes found._")
            lines.append("")
            continue
        lines.append(
            "| kind | axis | n_h | n_o | mean_hack | mean_honest | diff | d | p | q | AUC |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in hits[:20]:
            lines.append(
                f"| {r['axis_kind']} | `{r['axis_name']}` | "
                f"{r['n_hack']} | {r['n_honest']} | "
                f"{r['mean_hack']:+.3f} | {r['mean_honest']:+.3f} | "
                f"{r['mean_diff']:+.3f} | {r['cohens_d']:+.2f} | "
                f"{r['p']:.3g} | {r['q']:.3g} | {r['auc']:.2f} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Per-session score = mean projection across all decision-token positions "
        "(`tool_cmd_first`, `think_end`, `hack_kw`, `fix_kw`, `visible_*`, etc.)."
    )
    lines.append(
        "- Welch t-test is used because hack and honest pools have different "
        "variances and small sizes. p-values are two-sided."
    )
    lines.append(
        "- BH-FDR `q` is computed across all axes evaluated. Anything with q<0.10 "
        "would be a real signal; with N this small, expect mostly NaN/insignificant."
    )
    lines.append(
        "- The 'a-priori clusters' are fixed substrings; we read them no matter "
        "their global rank, to avoid double-dipping when reporting headline axes."
    )
    out_path.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--projections-dir", default=str(DEFAULT_PROJ))
    p.add_argument("--results-csv", default=str(DEFAULT_RESULTS))
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument(
        "--framings",
        nargs="*",
        default=["primary", "inclusive"],
        choices=list(FRAMINGS.keys()),
    )
    args = p.parse_args()

    proj_dir = Path(args.projections_dir)
    out_prefix = (
        Path(args.out_prefix)
        if args.out_prefix
        else proj_dir / "axes_contrast"
    )

    rows = load_projections(proj_dir / "projections_by_session")
    labels = load_results(Path(args.results_csv))

    print(f"loaded {len(rows):,} (session, axis) projection rows")
    print(f"loaded {len(labels)} session labels")

    for framing in args.framings:
        print(f"\n== framing: {framing} ==")
        results = contrast(rows, labels, framing=framing)
        n_eval = sum(1 for r in results if r["n_hack"] >= 2 and r["n_honest"] >= 2)
        print(f"axes with both groups present: {n_eval} / {len(results)}")

        suffix = f".{framing}"
        out_md = out_prefix.with_suffix(f"{suffix}.md")
        render_report(results, labels, out_md, top_n=args.top_n, framing=framing)
        print(f"wrote {out_md}")

        out_json = out_prefix.with_suffix(f"{suffix}.json")
        out_json.write_text(json.dumps(results, indent=2))
        print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
