"""Research CLI with explicitly separated audit, selection, evaluation, and reporting."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, new_directory, read_json, write_json
from .data import audit
from .evaluation import develop, evaluate
from .reporting import build_report, plot_eeg


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Personalized SVD + RBF EEG research; no automatic downloads")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("audit-data", "validate", "evaluate", "build-report", "plot-eeg"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--out", required=True, type=Path, help="New output directory; must not exist")
        if name in ("audit-data", "plot-eeg"):
            cmd.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
        if name == "validate":
            cmd.add_argument("--manifest", required=True, type=Path)
        if name == "evaluate":
            cmd.add_argument("--validation", required=True, type=Path)
        if name == "build-report":
            cmd.add_argument("--experiment", required=True, type=Path)
        if name == "plot-eeg":
            cmd.add_argument("--subject", required=True, type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        out = new_directory(args.out)
        if args.command == "audit-data":
            manifest = audit(load_config(args.config))
            write_json(out / "manifest.json", manifest)
            print(f"Eligible: {len(manifest['eligible_subjects'])}; excluded: {len(manifest['exclusions'])}")
            if not manifest["eligible_subjects"]:
                raise ValueError("No eligible subjects; exclusions saved in manifest.json")
        elif args.command == "validate":
            develop(read_json(args.manifest), out)
        elif args.command == "evaluate":
            evaluate(args.validation, out)
        elif args.command == "build-report":
            build_report(args.experiment, out)
        elif args.command == "plot-eeg":
            plot_eeg(load_config(args.config), args.subject, out)
        print(f"Completed {args.command}: {out}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
