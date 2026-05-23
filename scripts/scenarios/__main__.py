"""CLI for the scenario registry.

Usage:
  python -m scripts.scenarios list
  python -m scripts.scenarios bootstrap <name> <target_dir>
  python -m scripts.scenarios validate            # baseline pytest sanity check
  python -m scripts.scenarios validate <name>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import all_names, all_scenarios, get


def cmd_list(_: argparse.Namespace) -> int:
    for s in all_scenarios():
        print(
            f"{s.name:18s}  domain={s.domain:18s}  "
            f"baseline {s.baseline_pass}/{s.baseline_total} pass  "
            f"({len(s.bugs)} bugs)  {s.description}"
        )
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    s = get(args.name)
    target = Path(args.target).expanduser().resolve()
    if target.exists() and not args.force:
        print(f"refusing to overwrite {target} (use --force)", file=sys.stderr)
        return 2
    if target.exists():
        shutil.rmtree(target)
    s.write_to(target)
    print(f"bootstrapped {s.name} -> {target}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    names = [args.name] if args.name else all_names()
    failed = 0

    for name in names:
        s = get(name)
        with tempfile.TemporaryDirectory(prefix=f"leash-validate-{name}-") as tmp:
            target = Path(tmp) / "demo"
            s.write_to(target)
            source = (target / s.source_file).read_text()

            present = s.bugs_present(source)
            missing = [b.name for b in s.bugs if b.name not in present]

            proc = subprocess.run(
                ["pytest", "--tb=no", "-q"],
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=120,
            )
            tail = proc.stdout.strip().splitlines()
            summary_line = tail[-1] if tail else ""

            n_pass = _count(proc.stdout, " passed")
            n_fail = _count(proc.stdout, " failed")

            ok_baseline = (n_pass == s.baseline_pass) and (n_fail == s.baseline_fail)
            ok_bugs = len(missing) == 0
            ok = ok_baseline and ok_bugs

            tag = "OK " if ok else "BAD"
            print(
                f"[{tag}] {name:18s}  pytest: {n_pass}p/{n_fail}f "
                f"(expected {s.baseline_pass}p/{s.baseline_fail}f)  "
                f"bugs detected: {len(present)}/{len(s.bugs)}"
            )
            if not ok_bugs:
                print(f"        missing bug detectors: {missing}")
            if not ok_baseline:
                print(f"        pytest summary: {summary_line!r}")
            if not ok:
                failed += 1

    return 0 if failed == 0 else 1


def _count(output: str, suffix: str) -> int:
    """Crude count of '<n>{suffix}' in pytest output's summary line."""
    import re
    m = re.search(r"(\d+)" + re.escape(suffix), output)
    return int(m.group(1)) if m else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scenarios")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list all scenarios")
    p_list.set_defaults(func=cmd_list)

    p_boot = sub.add_parser("bootstrap", help="write a scenario into a target dir")
    p_boot.add_argument("name")
    p_boot.add_argument("target")
    p_boot.add_argument("--force", action="store_true",
                        help="overwrite the target dir if it exists")
    p_boot.set_defaults(func=cmd_bootstrap)

    p_val = sub.add_parser("validate", help="run pytest baseline + bug-detector check")
    p_val.add_argument("name", nargs="?")
    p_val.set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
