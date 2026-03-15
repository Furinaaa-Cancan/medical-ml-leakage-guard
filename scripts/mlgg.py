#!/usr/bin/env python3
"""
Unified CLI entrypoint for ml-leakage-guard.

This is a thin wrapper that forwards subcommands to existing scripts, so users
can use one stable command surface in terminal workflows and agent automation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

# Subprocess timeout: gates can be slow (large datasets) but should not run forever.
# Per-subprocess wall-clock limit. Adjust via MLGG_SUBPROCESS_TIMEOUT env var.
_DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = int(
    os.environ.get("MLGG_SUBPROCESS_TIMEOUT", "3600")
)

# Allowed --python executable names (basenames only).  Full-path executables
# matching these basenames are also accepted.
_ALLOWED_PYTHON_BASENAMES = frozenset(
    [
        "python", "python3", "python3.8", "python3.9", "python3.10",
        "python3.11", "python3.12", "python3.13",
    ]
)

# Maximum allowed byte-length for any single CLI argument to prevent
# memory exhaustion from absurdly long strings.
_MAX_ARG_BYTES = 8192


def _validate_python_bin(value: str) -> str:
    """
    Validate that --python points to a Python-like executable.

    Accepts:
    - sys.executable (always trusted)
    - Any path whose basename is in _ALLOWED_PYTHON_BASENAMES
    - Any path found via shutil.which that resolves to a python binary

    Raises SystemExit(2) with an informative message on failure.
    """
    if not value or not value.strip():
        return sys.executable
    candidate = value.strip()
    if candidate == sys.executable:
        return candidate
    import shutil
    basename = Path(candidate).name.lower()
    # Strip .exe suffix on Windows
    if basename.endswith(".exe"):
        basename = basename[:-4]
    if basename not in _ALLOWED_PYTHON_BASENAMES:
        print(
            f"[FAIL] invalid_python_executable: --python '{candidate}' is not a recognized "
            f"Python executable. Allowed basenames: {sorted(_ALLOWED_PYTHON_BASENAMES)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # Ensure it actually exists / is findable
    resolved = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
    if resolved is None:
        print(
            f"[FAIL] python_executable_not_found: --python '{candidate}' not found on PATH "
            f"or filesystem.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return resolved


# Forbidden path prefixes reused from _gate_utils._FORBIDDEN_PATH_PREFIXES.
# Duplicated here to avoid importing from scripts/ at module load time (circular risk).
_FORBIDDEN_CWD_PREFIXES = frozenset(
    ["/etc", "/private/etc", "/proc", "/sys", "/dev", "/var/run", "/boot", "/sbin"]
)

# Profile name must be a safe identifier: alphanumeric, hyphens, underscores only.
_PROFILE_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')


def _validate_cwd(value: str) -> Path:
    """
    Validate that --cwd is an existing directory, not a forbidden system path.

    Prevents:
    - Path traversal to /etc, /proc, /sys, etc.
    - Non-existent or file paths being used as working directories.
    """
    if "\x00" in value:
        print("[FAIL] invalid_cwd: --cwd contains NUL byte.", file=sys.stderr)
        raise SystemExit(2)
    try:
        cwd = Path(value).expanduser().resolve()
    except Exception as exc:
        print(f"[FAIL] invalid_cwd: cannot resolve --cwd '{value}': {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    # Forbidden system path check
    cwd_str = str(cwd)
    for prefix in _FORBIDDEN_CWD_PREFIXES:
        if cwd_str == prefix or cwd_str.startswith(prefix + "/"):
            print(
                f"[FAIL] cwd_forbidden_path: --cwd '{cwd}' targets a forbidden system "
                f"location. Forbidden prefixes: {sorted(_FORBIDDEN_CWD_PREFIXES)}",
                file=sys.stderr,
            )
            raise SystemExit(2)
    if not cwd.exists():
        print(f"[FAIL] cwd_not_found: --cwd directory does not exist: {cwd}", file=sys.stderr)
        raise SystemExit(2)
    if not cwd.is_dir():
        print(f"[FAIL] cwd_not_directory: --cwd path is not a directory: {cwd}", file=sys.stderr)
        raise SystemExit(2)
    return cwd


def _validate_profile_name(value: str) -> str:
    """
    Validate --profile-name is a safe identifier.

    Rejects values containing path separators (/ \\), null bytes, or characters
    that could be used for path traversal when the name is used as a filename.
    Allowed: alphanumeric, hyphens, underscores, 1-128 characters.
    """
    if not value or not value.strip():
        return ""
    cleaned = value.strip()
    if not _PROFILE_NAME_RE.match(cleaned):
        print(
            f"[FAIL] invalid_profile_name: --profile-name '{cleaned}' contains invalid "
            f"characters. Allowed: alphanumeric, hyphens, underscores (max 128 chars).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return cleaned


def _validate_passthrough(passthrough: List[str]) -> List[str]:
    """
    Validate pass-through arguments for basic safety.

    Checks:
    - Each token length ≤ _MAX_ARG_BYTES (prevents memory exhaustion)
    - No NUL bytes (prevent arg smuggling on some systems)

    Does NOT block specific flags — that's the subcommand's responsibility.
    """
    validated: List[str] = []
    for i, token in enumerate(passthrough):
        if len(token.encode("utf-8", errors="replace")) > _MAX_ARG_BYTES:
            print(
                f"[FAIL] passthrough_arg_too_long: argument at index {i} exceeds "
                f"{_MAX_ARG_BYTES} bytes. Possible memory exhaustion attempt.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if "\x00" in token:
            print(
                f"[FAIL] passthrough_arg_nul_byte: argument at index {i} contains a NUL "
                f"byte, which is not allowed.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        validated.append(token)
    return validated


def _run_subprocess(
    cmd: List[str],
    cwd: Path,
    timeout: Optional[int] = None,
) -> int:
    """
    Centralized subprocess launcher with timeout and error handling.

    All subprocess.run calls in this module should go through this function
    to ensure consistent timeout enforcement and return-code handling.
    """
    effective_timeout = timeout if timeout is not None else _DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), text=True, timeout=effective_timeout)
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        print(
            f"[FAIL] subprocess_timeout: command exceeded {effective_timeout}s timeout. "
            f"Set MLGG_SUBPROCESS_TIMEOUT env var to increase limit.",
            file=sys.stderr,
        )
        return 2
    except FileNotFoundError as exc:
        print(f"[FAIL] subprocess_not_found: {exc}", file=sys.stderr)
        return 2
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
    "split": (SCRIPTS_ROOT / "split_data.py", "Split a single CSV into train/valid/test with medical safety guarantees."),
    "doctor": (SCRIPTS_ROOT / "env_doctor.py", "Check runtime dependencies and optional backends."),
    "preflight": (SCRIPTS_ROOT / "schema_preflight.py", "Validate train/valid/test schema and semantic mapping."),
    "workflow": (SCRIPTS_ROOT / "run_productized_workflow.py", "Run doctor -> preflight -> strict -> summary."),
    "strict": (SCRIPTS_ROOT / "run_dag_pipeline.py", "Run strict fail-closed DAG gate pipeline."),
    "strict-legacy": (SCRIPTS_ROOT / "run_strict_pipeline.py", "Run legacy sequential strict pipeline (deprecated)."),
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
    "play": (
        SCRIPTS_ROOT / "mlgg_pixel.py",
        "Launch pixel-art interactive CLI launcher (guided menu experience).",
    ),
    "audit": (
        SCRIPTS_ROOT / "audit_external_project.py",
        "Quantitative 10-dimension audit of a medical ML project (100-point scale).",
    ),
    "fairness": (
        SCRIPTS_ROOT / "fairness_equity_gate.py",
        "Validate subgroup fairness and equity metrics (equalized odds, disparate impact).",
    ),
    "sample-size": (
        SCRIPTS_ROOT / "sample_size_gate.py",
        "Validate sample size adequacy (EPV, shrinkage factor, Riley criteria).",
    ),
    "batch-review": (
        SCRIPTS_ROOT / "batch_journal_review.py",
        "Batch audit N projects against journal standards with comparison matrix.",
    ),
    "audit-report": (
        SCRIPTS_ROOT / "generate_audit_report.py",
        "Generate comprehensive audit report with TRIPOD+AI/PROBAST+AI coverage, error KB lookup, and literature citations.",
    ),
    "export-review-prompt": (
        SCRIPTS_ROOT / "export_review_prompt.py",
        "Export MLGG review criteria as a portable LLM prompt. Users paste the output into any LLM (Claude, GPT-4, Gemini) to review a paper without local deployment.",
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


def _extract_option_value(argv: list[str], option: str, default: str) -> str:
    for idx, token in enumerate(argv):
        if token == option and idx + 1 < len(argv):
            value = str(argv[idx + 1]).strip()
            if value:
                return value
        if token.startswith(option + "="):
            value = token.split("=", 1)[1].strip()
            if value:
                return value
    return default


def _find_subcommand(argv: list[str]) -> tuple[int, str] | None:
    for idx, token in enumerate(argv):
        if token in COMMANDS:
            return idx, str(token)
    return None


def maybe_forward_subcommand_help(raw_argv: list[str]) -> int | None:
    """
    Forward intuitive help forms like:
    - mlgg.py onboarding --help
    - mlgg.py train --interactive --help
    - mlgg.py interactive --help
    """
    if not raw_argv:
        return None
    hit = _find_subcommand(raw_argv)
    if not hit:
        return None
    subcommand_index, subcommand = hit
    suffix = raw_argv[subcommand_index + 1 :]
    if not any(token in {"-h", "--help"} for token in suffix):
        return None
    # Keep explicit passthrough handling in normal parse flow.
    if "--" in suffix:
        return None

    python_bin = _validate_python_bin(
        _extract_option_value(raw_argv, "--python", sys.executable)
    )
    cwd_raw = _extract_option_value(raw_argv, "--cwd", str(REPO_ROOT))
    cwd = _validate_cwd(cwd_raw)

    interactive_requested = subcommand == "interactive" or "--interactive" in raw_argv
    if interactive_requested:
        wizard_script = COMMANDS["interactive"][0]
        if not wizard_script.exists():
            print(f"[FAIL] Interactive script not found: {wizard_script}", file=sys.stderr)
            return 2
        target_command = subcommand if subcommand in INTERACTIVE_CORE_COMMANDS else ""
        if subcommand == "interactive":
            target_command = _extract_option_value(raw_argv, "--command", "")
        cmd = [python_bin, str(wizard_script)]
        if target_command in INTERACTIVE_CORE_COMMANDS:
            cmd.extend(["--command", target_command])
        cmd.append("--help")
        print(f"$ {shlex.join(cmd)}")
        return _run_subprocess(cmd, cwd, timeout=60)

    script_path = COMMANDS[subcommand][0]
    if not script_path.exists():
        print(f"[FAIL] Script not found for command '{subcommand}': {script_path}", file=sys.stderr)
        return 2
    cmd = [python_bin, str(script_path), "--help"]
    print(f"$ {shlex.join(cmd)}")
    return _run_subprocess(cmd, cwd, timeout=60)


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
            "  python3 scripts/mlgg.py benchmark-suite --profile release --repeat 3 --emit-junit /tmp/mlgg_benchmark.junit.xml\n"
            "  python3 scripts/mlgg.py authority-release\n"
            "  python3 scripts/mlgg.py authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060\n"
            "  python3 scripts/mlgg.py play -- --strict-small-sample\n"
            "  python3 scripts/mlgg.py play -- --strict-small-sample --fail-on-play-blockers\n"
            "\n"
            "Tip:\n"
            "  Use `<subcommand> --help` for direct script help (e.g., `mlgg.py onboarding --help`).\n"
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
    forwarded_help = maybe_forward_subcommand_help(sys.argv[1:])
    if forwarded_help is not None:
        return int(forwarded_help)

    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    passthrough = [token for token in passthrough if token != "--"]

    subcommand = str(args.subcommand)
    python_bin = _validate_python_bin(str(args.python).strip() or sys.executable)
    cwd = _validate_cwd(str(args.cwd))
    passthrough = _validate_passthrough(passthrough)
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
            return _run_subprocess(cmd, cwd, timeout=60)
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
        profile_name = _validate_profile_name(str(args.profile_name))
        if profile_name:
            cmd.extend(["--profile-name", profile_name])
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
        return _run_subprocess(cmd, cwd)

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
    return _run_subprocess(cmd, cwd)


def cli_main() -> None:
    """Entry point for ``console_scripts`` (pyproject.toml)."""
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
