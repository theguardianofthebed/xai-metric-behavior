"""Command-line entry point; no experiment runs merely by importing the package."""
from __future__ import annotations

import argparse
import json

from .experiment import freeze_confirmation, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas XAI experiments: population FC and finite-sampling behavior")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Execute smoke/development or a previously frozen confirmation")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True, help="New output directory; existing directories are never overwritten")
    run.add_argument("--freeze", help="Required only for the confirmation phase")
    freeze = commands.add_parser("freeze", help="Freeze confirmation after inspecting development")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--development-run", required=True)
    freeze.add_argument("--output", required=True, help="New JSON file; never overwrites a prior freeze")
    real = commands.add_parser("run-real", help="EXP-01-R: Adult + Captum IG + official Quantus smoke/development")
    real.add_argument("--config", required=True)
    real.add_argument("--output", required=True, help="New output directory; existing results are preserved")
    real.add_argument("--data-dir", default="data/cache/openxai-adult", help="Cache for hash-verified author dataset files")
    args = parser.parse_args()
    if args.command == "run-real":
        from .exp01r import run_exp01r
        result = run_exp01r(args.config, args.output, args.data_dir)
        print(json.dumps({"status": result["status"], "phase": result["phase"],
                          "run_id": result["run_id"], "output": args.output,
                          "counts": result["counts"], "validation": result["validation"]}, indent=2))
        return
    if args.command == "run":
        result = run_experiment(args.config, args.output, freeze_path=args.freeze)
        print(json.dumps({"status": result["status"], "phase": result["phase"],
                          "run_id": result["run_id"], "output": args.output,
                          "counts": result["counts"], "validation": result["validation"]}, indent=2))
    else:
        result = freeze_confirmation(args.config, args.development_run, args.output)
        print(json.dumps({"status": result["status"], "output": args.output,
                          "config_sha256": result["config_sha256"],
                          "implementation_sha256": result["implementation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
