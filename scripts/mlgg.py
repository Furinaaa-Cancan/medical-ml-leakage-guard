#!/usr/bin/env python3
"""
Unified CLI entrypoint for ml-leakage-guard.

This is a thin wrapper that forwards subcommands to existing scripts, so users
can use one stable command surface in terminal workflows and agent automation.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments" / "authority-e2e"


COMMANDS: Dict[str, Tuple[Path, str]] = {
    "init": (SCRIPTS_ROOT / "init_project.py", "Initialize project folders and config templates."),
    "doctor": (SCRIPTS_ROOT / "env_doctor.py", "Check runtime dependencies and optional backends."),
    "preflight": (SCRIPTS_ROOT / "schema_preflight.py", "Validate train/valid/test schema and semantic mapping."),
    "workflow": (SCRIPTS_ROOT / "run_productized_workflow.py", "Run doctor -> preflight -> strict -> summary."),
    "strict": (SCRIPTS_ROOT / "run_strict_pipeline.py", "Run strict fail-closed gate pipeline."),
    "summary": (SCRIPTS_ROOT / "render_user_summary.py", "Render user-facing markdown/json summary."),
    "train": (SCRIPTS_ROOT / "train_select_evaluate.py", "Train/select/evaluate and emit evidence artifacts."),
    "authority": (EXPERIMENTS_ROOT / "run_authority_e2e.py", "Run authority E2E benchmark suite."),
    "scan-diabetes": (
        EXPERIMENTS_ROOT / "scan_stress_diabetes_feasibility.py",
        "Scan stress-case diabetes feasibility across target modes and row caps.",
    ),
    "adversarial": (
        EXPERIMENTS_ROOT / "run_adversarial_gate_checks.py",
        "Run adversarial fail-closed gate scenarios.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    command_help = "\n".join([f"  - {name}: {desc}" for name, (_, desc) in sorted(COMMANDS.items())])
    parser = argparse.ArgumentParser(
        description=(
            "ml-leakage-guard unified CLI.\n\n"
            "Available commands:\n"
            f"{command_help}\n\n"
            "Examples:\n"
            "  python3 scripts/mlgg.py init --project-root /tmp/mlgg_demo\n"
            "  python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict\n"
            "  python3 scripts/mlgg.py authority --include-stress-cases\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("command", choices=sorted(COMMANDS.keys()), help="Subcommand to execute.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run underlying scripts (default: current interpreter).",
    )
    parser.add_argument(
        "--cwd",
        default=str(REPO_ROOT),
        help="Working directory for the subcommand (default: repository root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the forwarded command and exit without executing.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    script_path, _ = COMMANDS[str(args.command)]
    if not script_path.exists():
        print(f"[FAIL] Script not found for command '{args.command}': {script_path}", file=sys.stderr)
        return 2

    python_bin = str(args.python).strip() or sys.executable
    cwd = Path(str(args.cwd)).expanduser().resolve()
    cmd = [python_bin, str(script_path), *passthrough]

    print(f"$ {shlex.join(cmd)}")
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

