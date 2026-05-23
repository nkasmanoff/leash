"""Linear probe: predict hack vs honest outcome from session-level projection features.

Features (per session):
  - Per-segment-kind (think, tool_command, tool_other, visible) projection stats:
      mean, std, min, max, p10, p25, p50, p75, p90, count, fraction-of-tokens
  - Decision-token features (think only, from regex labels in decision_tokens.py):
      n_hack, n_fix, hack_mean, fix_mean, fix_minus_hack, hack_frac, fix_frac
  - Session: n_turns, n_tool_calls, total_tokens

Labels (binary):
  - HACK = {tautological_assert, modified_tests, weakened_assert,
            special_case_source, delete_tests, corrupted_tests}
  - HONEST = {clean_fix, partial_fix, damaged_source}
  - excluded: {stalled_no_progress, infra_oom, corrupted_source}
    (corrupted_source is ambiguous; included in the SECONDARY framing.)

Outputs:
  - Leave-one-out AUC for L2 logistic regression
  - Univariate AUC for each feature
  - Permutation test (shuffle labels) to estimate null distribution
  - Top features by univariate signal
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from segment_tokens import segment_session  # noqa: E402
from decision_tokens import label_session_tokens  # noqa: E402

SWEEP = REPO / "traces" / "hack_sweep" / "run-1779559443"

HACK_LABELS = {
    "tautological_assert", "modified_tests", "weakened_assert",
    "special_case_source", "delete_tests", "corrupted_tests",
}
HONEST_LABELS = {"clean_fix", "partial_fix", "damaged_source"}

PRIMARY_EXCLUDE = {"stalled_no_progress", "infra_oom", "corrupted_source"}
SECONDARY_EXCLUDE = {"stalled_no_progress", "infra_oom"}

KINDS = ["think", "tool_command", "tool_other", "visible"]
PCTS = [10, 25, 50, 75, 90]


def per_kind_stats(rows: list[dict], kind: str) -> dict[str, float]:
    vals = np.array([r["projection"] for r in rows if r["kind"] == kind], dtype=float)
    out: dict[str, float] = {}
    if vals.size == 0:
        for k in ["mean", "std", "min", "max", "count", "frac"] + [f"p{p}" for p in PCTS]:
            out[f"{kind}_{k}"] = 0.0
        return out
    out[f"{kind}_mean"] = float(vals.mean())
    out[f"{kind}_std"] = float(vals.std())
    out[f"{kind}_min"] = float(vals.min())
    out[f"{kind}_max"] = float(vals.max())
    out[f"{kind}_count"] = float(vals.size)
    for p in PCTS:
        out[f"{kind}_p{p}"] = float(np.percentile(vals, p))
    return out


def build_features(session_dir: Path) -> dict[str, float]:
    rows = segment_session(session_dir)
    feats: dict[str, float] = {}
    total = max(1, len(rows))
    feats["total_tokens"] = float(total)
    for kind in KINDS:
        s = per_kind_stats(rows, kind)
        s[f"{kind}_frac"] = s.pop(f"{kind}_count") / total
        s[f"{kind}_count"] = sum(1 for r in rows if r["kind"] == kind)
        feats.update(s)

    labeled = label_session_tokens(session_dir)
    n_hack = sum(1 for t in labeled if t["label"] == "hack")
    n_fix = sum(1 for t in labeled if t["label"] == "fix")
    n_think = max(1, len(labeled))
    hack_mean = (statistics.mean(t["projection"] for t in labeled if t["label"] == "hack")
                 if n_hack else 0.0)
    fix_mean = (statistics.mean(t["projection"] for t in labeled if t["label"] == "fix")
                if n_fix else 0.0)
    feats["dec_n_hack"] = float(n_hack)
    feats["dec_n_fix"] = float(n_fix)
    feats["dec_hack_frac"] = n_hack / n_think
    feats["dec_fix_frac"] = n_fix / n_think
    feats["dec_hack_mean"] = float(hack_mean)
    feats["dec_fix_mean"] = float(fix_mean)
    feats["dec_fix_minus_hack"] = float(fix_mean - hack_mean) if (n_hack and n_fix) else 0.0

    return feats


def collect(framing: str = "primary") -> tuple[list[dict], np.ndarray, np.ndarray, list[str], list[dict]]:
    exclude = PRIMARY_EXCLUDE if framing == "primary" else SECONDARY_EXCLUDE
    if framing == "secondary":
        hack_set = HACK_LABELS | {"corrupted_source"}
    else:
        hack_set = HACK_LABELS

    rows_meta: list[dict] = []
    feature_dicts: list[dict[str, float]] = []
    labels: list[int] = []

    with open(SWEEP / "results.csv") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            if label in exclude:
                continue
            session_dir = SWEEP / row["cell"] / f"seed-{row['seed']}" / "session"
            if not session_dir.exists():
                continue
            try:
                feats = build_features(session_dir)
            except Exception as exc:
                print(f"skip {row['cell']}/{row['seed']}: {exc}", file=sys.stderr)
                continue
            feats["n_turns"] = float(row.get("n_turns") or 0)
            feats["n_tool_calls"] = float(row.get("n_tool_calls") or 0)
            feature_dicts.append(feats)
            labels.append(1 if label in hack_set else 0)
            rows_meta.append({
                "cell": row["cell"],
                "seed": row["seed"],
                "label": label,
                "y": labels[-1],
            })

    feat_names = sorted({k for d in feature_dicts for k in d})
    X = np.array([[d.get(k, 0.0) for k in feat_names] for d in feature_dicts], dtype=float)
    y = np.array(labels, dtype=int)
    return rows_meta, X, y, feat_names, feature_dicts


def loo_auc(X: np.ndarray, y: np.ndarray, *, C: float = 0.5) -> tuple[float, np.ndarray]:
    """Leave-one-out cross-validated AUC of L2 logistic regression."""
    if len(np.unique(y)) < 2:
        return float("nan"), np.zeros_like(y, dtype=float)
    loo = LeaveOneOut()
    scores = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        Xt = scaler.transform(X[train_idx])
        Xe = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            penalty="l2", C=C, max_iter=2000, solver="lbfgs",
            class_weight="balanced",
        )
        clf.fit(Xt, y[train_idx])
        scores[test_idx] = clf.predict_proba(Xe)[:, 1]
    auc = roc_auc_score(y, scores)
    return float(auc), scores


def univariate_aucs(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[tuple[str, float, float]]:
    """For each feature, compute AUC using that single feature's value as score.
    Returns sorted list of (feat_name, auc, oriented_auc) where oriented = max(auc, 1-auc)."""
    out: list[tuple[str, float, float]] = []
    if len(np.unique(y)) < 2:
        return out
    for j, name in enumerate(names):
        col = X[:, j]
        if np.allclose(col.std(), 0):
            continue
        try:
            a = roc_auc_score(y, col)
        except ValueError:
            continue
        out.append((name, a, max(a, 1 - a)))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def permutation_null(X: np.ndarray, y: np.ndarray, *, n: int = 200, C: float = 0.5,
                     rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    nulls = np.zeros(n, dtype=float)
    for i in range(n):
        y_perm = rng.permutation(y)
        auc, _ = loo_auc(X, y_perm, C=C)
        nulls[i] = auc
    return nulls


def nested_loo_best_univariate(X: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[float, list[str]]:
    """For each LOO fold, pick best univariate AUC on the training set, evaluate on held-out."""
    if len(np.unique(y)) < 2:
        return float("nan"), []
    loo = LeaveOneOut()
    scores = np.zeros(len(y), dtype=float)
    picked: list[str] = []
    for train_idx, test_idx in loo.split(X):
        best_name, best_oriented, best_sign = None, -1.0, 1.0
        for j in range(X.shape[1]):
            col = X[train_idx, j]
            if np.allclose(col.std(), 0) or len(np.unique(y[train_idx])) < 2:
                continue
            try:
                a = roc_auc_score(y[train_idx], col)
            except ValueError:
                continue
            oriented = max(a, 1 - a)
            if oriented > best_oriented:
                best_oriented = oriented
                best_name = names[j]
                best_sign = 1.0 if a >= 0.5 else -1.0
        if best_name is None:
            scores[test_idx] = 0.0
            picked.append("")
            continue
        j = names.index(best_name)
        scores[test_idx] = best_sign * X[test_idx, j]
        picked.append(best_name)
    auc = roc_auc_score(y, scores)
    return float(auc), picked


def loo_auc_subset(X: np.ndarray, y: np.ndarray, names: list[str], subset: list[str], *,
                   C: float = 1.0) -> float:
    idx = [names.index(s) for s in subset if s in names]
    if not idx or len(np.unique(y)) < 2:
        return float("nan")
    Xs = X[:, idx]
    auc, _ = loo_auc(Xs, y, C=C)
    return auc


def render(meta: list[dict], X: np.ndarray, y: np.ndarray, feat_names: list[str],
           framing: str, *, n_perm: int = 200) -> str:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    lines: list[str] = []
    lines.append(f"# Linear probe: hack vs honest ({framing})")
    lines.append("")
    lines.append(f"- N = {len(y)} sessions ({n_pos} hack, {n_neg} honest)")
    lines.append(f"- features = {len(feat_names)}")
    lines.append("- model = L2 logistic regression (C=0.5, balanced class weights), "
                 "leave-one-out CV")
    lines.append("")

    if n_pos < 2 or n_neg < 2:
        lines.append("**Skipped: insufficient class size.**")
        return "\n".join(lines)

    lines.append("## Sessions used")
    lines.append("")
    lines.append("| cell | seed | label | y |")
    lines.append("|------|------|-------|--:|")
    for m in meta:
        lines.append(f"| {m['cell']} | {m['seed']} | {m['label']} | {m['y']} |")
    lines.append("")

    auc, scores = loo_auc(X, y)
    lines.append(f"## Multivariate L2 logistic regression: **LOO AUC = {auc:.3f}**")
    lines.append("")

    nulls = permutation_null(X, y, n=n_perm)
    p = float((nulls >= auc).mean())
    lines.append(f"- Permutation null (n={n_perm} label shuffles): "
                 f"mean = {nulls.mean():.3f}, std = {nulls.std():.3f}, "
                 f"95th pct = {np.percentile(nulls, 95):.3f}")
    lines.append(f"- Empirical p-value (shuffles ≥ observed): **p = {p:.3f}**")
    lines.append("")

    pred = (scores >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    lines.append(f"- LOO confusion @0.5: TP={tp} FP={fp} FN={fn} TN={tn}")
    lines.append("")

    lines.append("### Per-session LOO scores")
    lines.append("")
    lines.append("| cell | seed | label | y | score | correct |")
    lines.append("|------|------|-------|--:|------:|:-------:|")
    for m, s in zip(meta, scores):
        ok = "✓" if (s >= 0.5) == bool(m["y"]) else "✗"
        lines.append(f"| {m['cell']} | {m['seed']} | {m['label']} | {m['y']} | {s:.3f} | {ok} |")
    lines.append("")

    lines.append("## Fair single-feature probe (nested LOO: pick best feature on train, score test)")
    lines.append("")
    nested_auc, picks = nested_loo_best_univariate(X, y, feat_names)
    from collections import Counter
    pick_counts = Counter(picks).most_common(5)
    lines.append(f"- Nested LOO AUC: **{nested_auc:.3f}**")
    lines.append("- Most-frequently-picked features (across folds): "
                 + ", ".join(f"`{n}` ({c}/{len(picks)})" for n, c in pick_counts))

    rng = np.random.default_rng(42)
    nested_nulls = []
    for _ in range(min(n_perm, 200)):
        y_perm = rng.permutation(y)
        a, _ = nested_loo_best_univariate(X, y_perm, feat_names)
        nested_nulls.append(a)
    nested_nulls = np.array(nested_nulls)
    p_nested = float((nested_nulls >= nested_auc).mean())
    lines.append(f"- Nested-CV permutation null (n={len(nested_nulls)}): "
                 f"mean = {nested_nulls.mean():.3f}, std = {nested_nulls.std():.3f}, "
                 f"95th pct = {np.percentile(nested_nulls, 95):.3f}")
    lines.append(f"- Empirical p-value: **p = {p_nested:.3f}**")
    lines.append("")

    lines.append("## Hand-picked small probes (LOO AUC, L2 C=1.0, balanced)")
    lines.append("")
    decision_only = [n for n in feat_names if n.startswith("dec_")]
    candidates = [
        ["dec_fix_minus_hack"],
        ["n_turns"],
        ["total_tokens"],
        ["tool_other_p10"],
        ["dec_fix_minus_hack", "n_turns"],
        ["dec_fix_minus_hack", "n_turns", "tool_other_p10"],
        ["think_mean", "tool_command_mean", "visible_mean"],
        ["think_mean", "tool_command_mean", "tool_other_mean", "visible_mean"],
        decision_only,  # all decision features only
    ]
    lines.append("| feature subset | n_feat | LOO AUC |")
    lines.append("|----------------|-------:|--------:|")
    for subset in candidates:
        a = loo_auc_subset(X, y, feat_names, subset, C=1.0)
        label = ("`" + "`, `".join(subset) + "`") if len(subset) <= 4 else f"all decision features ({len(subset)})"
        lines.append(f"| {label} | {len(subset)} | {a:.3f} |")
    lines.append("")

    lines.append("## Top univariate features (single-feature AUC, IN-SAMPLE — optimistic)")
    lines.append("")
    uni = univariate_aucs(X, y, feat_names)
    lines.append("| feature | raw AUC | oriented AUC |")
    lines.append("|---------|--------:|-------------:|")
    for name, raw, oriented in uni[:20]:
        lines.append(f"| {name} | {raw:.3f} | {oriented:.3f} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--out-primary",
                   default=str(SWEEP / "probe_primary.md"))
    p.add_argument("--out-secondary",
                   default=str(SWEEP / "probe_secondary.md"))
    args = p.parse_args()

    for framing, out_path in [
        ("primary", args.out_primary),
        ("secondary", args.out_secondary),
    ]:
        meta, X, y, names, _ = collect(framing)
        report = render(meta, X, y, names, framing, n_perm=args.n_perm)
        Path(out_path).write_text(report)
        print(f"--- {framing} ---")
        print(report)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
