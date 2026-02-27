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
    "onboarding": (
        SCRIPTS_ROOT / "mlgg_onboarding.py",
        "Run guided novice onboarding (demo data -> train -> attestation -> strict workflow).",
    ),
    "interactive": (
        SCRIPTS_ROOT / "mlgg_interactive.py",
        "Launch interactive wizard for core commands (init/workflow/train/authority).",
    ),
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
INTERACTIVE_CORE_COMMANDS = ("init", "workflow", "train", "authority")


def build_parser() -> argparse.ArgumentParser:
    command_help = "\n".join([f"  - {name}: {desc}" for name, (_, desc) in sorted(COMMANDS.items())])
    parser = argparse.ArgumentParser(
        description=(
            "ml-leakage-guard unified CLI.\n\n"
            "Available commands:\n"
            f"{command_help}\n\n"
            "Examples:\n"
            "  python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode guided --yes\n"
            "  python3 scripts/mlgg.py init --project-root /tmp/mlgg_demo\n"
            "  python3 scripts/mlgg.py train --interactive\n"
            "  python3 scripts/mlgg.py interactive --command workflow\n"
            "  python3 scripts/mlgg.py interactive --command train --load-profile --profile-name demo --accept-defaults\n"
            "  python3 scripts/mlgg.py interactive --command train -- --help\n"
            "  python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict --allow-missing-compare\n"
            "  python3 scripts/mlgg.py workflow -- --help\n"
            "  python3 scripts/mlgg.py authority --include-stress-cases\n"
            "\n"
            "Tip:\n"
            "  Use `-- --help` to view subcommand-native help.\n"
            "  For interactive mode, include `--command` before `-- --help`.\n"
            "  Example: `python3 scripts/mlgg.py workflow -- --help`\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("subcommand", choices=sorted(COMMANDS.keys()), help="Subcommand to execute.")
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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run selected core command via interactive wizard (init/workflow/train/authority).",
    )
    parser.add_argument(
        "--command",
        dest="interactive_command",
        choices=list(INTERACTIVE_CORE_COMMANDS),
        help="Wizard target command when using subcommand=interactive.",
    )
    parser.add_argument(
        "--profile-name",
        default="",
        help="Profile name for interactive mode.",
    )
    parser.add_argument(
        "--profile-dir",
        default="~/.mlgg/profiles",
        help="Profile directory for interactive mode.",
    )
    parser.add_argument(
        "--save-profile",
        action="store_true",
        help="Save interactive selections into profile.",
    )
    parser.add_argument(
        "--load-profile",
        action="store_true",
        help="Load interactive defaults from profile.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Interactive mode only: print generated command without execution.",
    )
    parser.add_argument(
        "--accept-defaults",
        action="store_true",
        help="Interactive mode only: auto-accept prompt defaults when available.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    subcommand = str(args.subcommand)
    python_bin = str(args.python).strip() or sys.executable
    cwd = Path(str(args.cwd)).expanduser().resolve()
    interactive_requested = bool(args.interactive) or subcommand == "interactive"

    if interactive_requested:
        if subcommand == "interactive" and not args.interactive_command and passthrough and passthrough[0] in {"-h", "--help"}:
            wizard_script = COMMANDS["interactive"][0]
            if not wizard_script.exists():
                print(f"[FAIL] Interactive script not found: {wizard_script}", file=sys.stderr)
                return 2
            cmd = [python_bin, str(wizard_script), "--help"]
            print(f"$ {shlex.join(cmd)}")
            if args.dry_run:
                return 0
            proc = subprocess.run(cmd, cwd=str(cwd), text=True)
            return int(proc.returncode)
        target_command = str(args.interactive_command).strip() if args.interactive_command else subcommand
        if target_command == "interactive":
            print(
                "[FAIL] interactive mode requires --command "
                f"({'|'.join(INTERACTIVE_CORE_COMMANDS)}).",
                file=sys.stderr,
            )
            return 2
        if target_command not in INTERACTIVE_CORE_COMMANDS:
            print(
                "[FAIL] --interactive is supported only for: "
                + ", ".join(INTERACTIVE_CORE_COMMANDS),
                file=sys.stderr,
            )
            return 2
        wizard_script = COMMANDS["interactive"][0]
        if not wizard_script.exists():
            print(f"[FAIL] Interactive script not found: {wizard_script}", file=sys.stderr)
            return 2
        cmd = [
            python_bin,
            str(wizard_script),
            "--command",
            target_command,
            "--python",
            python_bin,
            "--cwd",
            str(cwd),
            "--profile-dir",
            str(args.profile_dir),
        ]
        if str(args.profile_name).strip():
            cmd.extend(["--profile-name", str(args.profile_name).strip()])
        if args.save_profile:
            cmd.append("--save-profile")
        if args.load_profile:
            cmd.append("--load-profile")
        if args.print_only:
            cmd.append("--print-only")
        if args.accept_defaults:
            cmd.append("--accept-defaults")
        cmd.extend(passthrough)
        print(f"$ {shlex.join(cmd)}")
        if args.dry_run:
            return 0
        proc = subprocess.run(cmd, cwd=str(cwd), text=True)
        return int(proc.returncode)

    script_path, _ = COMMANDS[subcommand]
    if not script_path.exists():
        print(f"[FAIL] Script not found for command '{subcommand}': {script_path}", file=sys.stderr)
        return 2
    cmd = [python_bin, str(script_path), *passthrough]
    print(f"$ {shlex.join(cmd)}")
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
