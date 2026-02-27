#!/usr/bin/env python3
"""
Unified CLI entrypoint for ml-leakage-guard.

This is a thin wrapper that forwards subcommands to existing scripts, so users
can use one stable command surface in terminal workflows and agent automation.
"""

from __future__ import annotations

import argparse
import json
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
    "benchmark-suite": (
        EXPERIMENTS_ROOT / "run_release_benchmark_matrix.py",
        "Run structured multi-dataset stability benchmark matrix (authority + adversarial).",
    ),
    "authority-release": (
        EXPERIMENTS_ROOT / "run_authority_e2e.py",
        "Run authority E2E with recommended release-grade stress route (CKD).",
    ),
    "authority-research-heart": (
        EXPERIMENTS_ROOT / "run_authority_e2e.py",
        "Run authority E2E with heart research/high-pressure stress route.",
    ),
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
COMMAND_PRESETS: Dict[str, Tuple[str, ...]] = {
    "authority-release": (
        "--include-stress-cases",
        "--stress-case-id",
        "uci-chronic-kidney-disease",
    ),
    "authority-research-heart": (
        "--include-stress-cases",
        "--stress-case-id",
        "uci-heart-disease",
        "--stress-seed-search",
    ),
}
PRESET_BLOCKED_FLAGS: Dict[str, Tuple[str, ...]] = {
    # Keep wrapper semantics strict and auditable: these wrappers should not
    # allow callers to override the fixed stress route via passthrough flags.
    "authority-release": (
        "--include-stress-cases",
        "--stress-case-id",
        "--stress-seed-search",
        "--no-stress-seed-search",
    ),
    "authority-research-heart": (
        "--include-stress-cases",
        "--stress-case-id",
        "--stress-seed-search",
        "--no-stress-seed-search",
    ),
}
AUTHORITY_PRESET_ROUTE_OVERRIDE_FORBIDDEN = "authority_preset_route_override_forbidden"
MLGG_ERROR_CONTRACT_VERSION = "mlgg_error.v1"


def passthrough_contains_flag(passthrough: list[str], flag: str) -> bool:
    for token in passthrough:
        if token == flag:
            return True
        if token.startswith(flag + "="):
            return True
    return False


def emit_fail(
    *,
    code: str,
    message: str,
    error_json: bool,
    details: Dict[str, object] | None = None,
) -> int:
    print(f"[FAIL] {code}: {message}", file=sys.stderr)
    if error_json:
        payload = {
            "contract_version": MLGG_ERROR_CONTRACT_VERSION,
            "status": "fail",
            "code": code,
            "message": message,
            "details": details or {},
        }
        print(json.dumps(payload, ensure_ascii=True), file=sys.stderr)
    return 2


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
            "  python3 scripts/mlgg.py benchmark-suite --profile release\n"
            "  python3 scripts/mlgg.py authority-release\n"
            "  python3 scripts/mlgg.py authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060\n"
            "\n"
            "Tip:\n"
            "  Use `-- --help` to view subcommand-native help.\n"
            "  For interactive mode, include `--command` before `-- --help`.\n"
            "  Example: `python3 scripts/mlgg.py workflow -- --help`\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
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
    parser.add_argument(
        "--error-json",
        action="store_true",
        help="Emit machine-readable JSON error payloads on failure.",
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
                return emit_fail(
                    code="interactive_script_not_found",
                    message=f"Interactive script not found: {wizard_script}",
                    error_json=bool(args.error_json),
                )
            cmd = [python_bin, str(wizard_script), "--help"]
            print(f"$ {shlex.join(cmd)}")
            if args.dry_run:
                return 0
            proc = subprocess.run(cmd, cwd=str(cwd), text=True)
            return int(proc.returncode)
        target_command = str(args.interactive_command).strip() if args.interactive_command else subcommand
        if target_command == "interactive":
            return emit_fail(
                code="interactive_command_missing",
                message=(
                    "interactive mode requires --command "
                    f"({'|'.join(INTERACTIVE_CORE_COMMANDS)})."
                ),
                error_json=bool(args.error_json),
            )
        if target_command not in INTERACTIVE_CORE_COMMANDS:
            return emit_fail(
                code="interactive_command_not_supported",
                message="--interactive is supported only for: " + ", ".join(INTERACTIVE_CORE_COMMANDS),
                error_json=bool(args.error_json),
            )
        wizard_script = COMMANDS["interactive"][0]
        if not wizard_script.exists():
            return emit_fail(
                code="interactive_script_not_found",
                message=f"Interactive script not found: {wizard_script}",
                error_json=bool(args.error_json),
            )
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
        return emit_fail(
            code="subcommand_script_not_found",
            message=f"Script not found for command '{subcommand}': {script_path}",
            error_json=bool(args.error_json),
            details={"subcommand": subcommand, "script_path": str(script_path)},
        )
    if subcommand in PRESET_BLOCKED_FLAGS:
        blocked = [
            flag
            for flag in PRESET_BLOCKED_FLAGS[subcommand]
            if passthrough_contains_flag(passthrough, flag)
        ]
        if blocked:
            return emit_fail(
                code=AUTHORITY_PRESET_ROUTE_OVERRIDE_FORBIDDEN,
                message=(
                    "preset command does not allow overriding fixed route flags: "
                    + ", ".join(blocked)
                ),
                error_json=bool(args.error_json),
                details={"subcommand": subcommand, "blocked_flags": blocked},
            )

    preset_args = list(COMMAND_PRESETS.get(subcommand, ()))
    cmd = [python_bin, str(script_path), *preset_args, *passthrough]
    print(f"$ {shlex.join(cmd)}")
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
