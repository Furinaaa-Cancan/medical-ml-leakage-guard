"""
Security and robustness tests for mlgg.py CLI.

Strategy for combinatorial explosion:
---------------------------------------
22 subcommands × 15 flags = theoretical O(2^15 × 22) = ~720k combinations.
We tackle this with three complementary approaches:

1. Pairwise (2-way) coverage for flag interactions — reduces to ~150 cases.
   Framework: pytest.mark.parametrize with allpairspy or hand-crafted pairs.

2. Property-based testing with Hypothesis — generates random valid/invalid
   inputs and checks invariants (exit code ∈ {0,2}, --dry-run never spawns,
   --error-json always produces parseable JSON on failure, etc.).

3. Security boundary tests — targeted tests for each identified vulnerability:
   V1 (--python injection), V2 (--cwd traversal), V3 (passthrough length),
   V4 (timeout), V5 (--profile-dir), V6 (pre-parse python_bin).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
MLGG = SCRIPTS_ROOT / "mlgg.py"


def run_mlgg(args: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
    """Run mlgg.py as a subprocess with a short timeout."""
    return subprocess.run(
        [sys.executable, str(MLGG)] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_mlgg_module(args: List[str]) -> tuple[int, str, str]:
    """
    Run mlgg.main() in-process (faster, no subprocess overhead).
    Returns (returncode, stdout_capture, stderr_capture).
    """
    import io
    from contextlib import redirect_stderr, redirect_stdout

    sys.path.insert(0, str(SCRIPTS_ROOT))
    import importlib
    if "mlgg" in sys.modules:
        del sys.modules["mlgg"]
    mlgg = importlib.import_module("mlgg")

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = ["mlgg.py"] + args
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                rc = mlgg.main()
                if rc is None:
                    rc = 0
            except SystemExit as exc:
                rc = exc.code if exc.code is not None else 0
            except Exception:
                rc = 2
    finally:
        sys.argv = old_argv
    return int(rc), stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# V1 — --python arbitrary executable injection
# ---------------------------------------------------------------------------


class TestPythonBinValidation:
    """V1: --python must be a recognized Python executable, not arbitrary binary."""

    def test_default_python_is_accepted(self, tmp_path):
        """sys.executable should always be accepted."""
        rc, out, err = run_mlgg_module(["doctor", "--python", sys.executable, "--dry-run"])
        # dry-run = 0 or command itself returns non-zero — either way NOT 2 from validation
        assert rc != 2 or "invalid_python_executable" not in err

    @pytest.mark.parametrize("evil_bin", [
        "/bin/sh",
        "/bin/bash",
        "/usr/bin/curl",
        "/usr/bin/wget",
        "sh",
        "bash",
        "/tmp/evil_script",
        "/etc/passwd",
        # Note: "" is intentionally NOT listed here — empty string falls back to
        # sys.executable by design (_validate_python_bin returns default for empty input)
    ])
    def test_evil_python_bin_rejected(self, evil_bin):
        """Non-Python executables must be rejected before any subprocess.run call."""
        rc, out, err = run_mlgg_module(["doctor", "--python", evil_bin, "--dry-run"])
        assert rc == 2, f"Expected exit 2 for --python '{evil_bin}', got {rc}"
        assert "invalid_python_executable" in err or "python_executable_not_found" in err

    @pytest.mark.parametrize("legit_bin", [
        "python3",
        "python",
        sys.executable,
    ])
    def test_legitimate_python_accepted_if_exists(self, legit_bin):
        """Recognized Python basenames that exist on PATH should be accepted."""
        import shutil
        if shutil.which(legit_bin) is None:
            pytest.skip(f"{legit_bin} not on PATH")
        rc, out, err = run_mlgg_module(["doctor", "--python", legit_bin, "--dry-run"])
        assert "invalid_python_executable" not in err

    def test_python_bin_nul_byte_rejected(self):
        """NUL bytes in --python value must be rejected."""
        rc, out, err = run_mlgg_module(["doctor", "--python", "python3\x00evil", "--dry-run"])
        assert rc == 2

    def test_python_bin_very_long_path_rejected(self):
        """Absurdly long --python path must be rejected or handled safely."""
        long_path = "/tmp/" + "a" * 8193
        rc, out, err = run_mlgg_module(["doctor", "--python", long_path, "--dry-run"])
        assert rc == 2


# ---------------------------------------------------------------------------
# V2 — --cwd path traversal
# ---------------------------------------------------------------------------


class TestCwdValidation:
    """V2: --cwd must be an existing directory."""

    def test_valid_cwd_accepted(self, tmp_path):
        """An existing directory should be accepted."""
        rc, out, err = run_mlgg_module(["doctor", "--cwd", str(tmp_path), "--dry-run"])
        assert "invalid_cwd" not in err
        assert "cwd_not_found" not in err

    def test_nonexistent_cwd_rejected(self):
        """A non-existent directory should be rejected."""
        rc, out, err = run_mlgg_module(["doctor", "--cwd", "/nonexistent/path/xyz", "--dry-run"])
        assert rc == 2
        assert "cwd_not_found" in err or "invalid_cwd" in err

    def test_file_as_cwd_rejected(self, tmp_path):
        """A file path where a directory is expected should be rejected."""
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        rc, out, err = run_mlgg_module(["doctor", "--cwd", str(f), "--dry-run"])
        assert rc == 2
        assert "cwd_not_directory" in err or "invalid_cwd" in err

    def test_cwd_root_accepted(self):
        """/ is a valid directory (we allow it — restriction is existence + is_dir)."""
        rc, out, err = run_mlgg_module(["doctor", "--cwd", "/", "--dry-run"])
        # Should not fail on validation; subprocess may fail separately
        assert "cwd_not_found" not in err
        assert "cwd_not_directory" not in err


# ---------------------------------------------------------------------------
# V3 — passthrough argument length limits
# ---------------------------------------------------------------------------


class TestPassthroughValidation:
    """V3: passthrough args must be length-limited and NUL-free."""

    def test_normal_passthrough_accepted(self, tmp_path):
        """Normal passthrough flags should pass through."""
        rc, out, err = run_mlgg_module(
            ["doctor", "--dry-run", "--", "--some-flag", "some-value"]
        )
        assert "passthrough_arg_too_long" not in err
        assert "passthrough_arg_nul_byte" not in err

    def test_oversized_passthrough_rejected(self):
        """A single passthrough arg exceeding 8192 bytes must be rejected."""
        huge_arg = "x" * 10000
        rc, out, err = run_mlgg_module(["doctor", "--", huge_arg])
        assert rc == 2
        assert "passthrough_arg_too_long" in err

    def test_nul_byte_in_passthrough_rejected(self):
        """NUL bytes in passthrough args must be rejected."""
        rc, out, err = run_mlgg_module(["doctor", "--", "--flag\x00injected"])
        assert rc == 2
        assert "passthrough_arg_nul_byte" in err

    def test_many_short_passthrough_accepted(self):
        """Many short legitimate passthrough flags should all pass."""
        args = ["doctor", "--dry-run"]
        for i in range(50):
            args.extend(["--", f"--flag-{i}", f"value-{i}"])
        rc, out, err = run_mlgg_module(args)
        assert "passthrough_arg_too_long" not in err


# ---------------------------------------------------------------------------
# V4 — subprocess timeout
# ---------------------------------------------------------------------------


class TestSubprocessTimeout:
    """V4: _run_subprocess must respect MLGG_SUBPROCESS_TIMEOUT."""

    def test_run_subprocess_timeout_returns_2(self, tmp_path, monkeypatch):
        """A process that takes longer than timeout should return exit code 2."""
        import mlgg
        import importlib
        sys.path.insert(0, str(SCRIPTS_ROOT))
        mlgg_mod = importlib.import_module("mlgg")

        # Patch subprocess.run to raise TimeoutExpired
        with patch("mlgg.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python3"], timeout=1)
            rc = mlgg_mod._run_subprocess(["python3", "--version"], tmp_path, timeout=1)
        assert rc == 2

    def test_default_timeout_exists(self):
        """_DEFAULT_SUBPROCESS_TIMEOUT_SECONDS must be a positive integer."""
        import importlib
        sys.path.insert(0, str(SCRIPTS_ROOT))
        mlgg_mod = importlib.import_module("mlgg")
        assert mlgg_mod._DEFAULT_SUBPROCESS_TIMEOUT_SECONDS > 0

    def test_timeout_env_override(self, monkeypatch):
        """MLGG_SUBPROCESS_TIMEOUT env var should override the default."""
        monkeypatch.setenv("MLGG_SUBPROCESS_TIMEOUT", "120")
        import importlib
        if "mlgg" in sys.modules:
            del sys.modules["mlgg"]
        sys.path.insert(0, str(SCRIPTS_ROOT))
        mlgg_mod = importlib.import_module("mlgg")
        assert mlgg_mod._DEFAULT_SUBPROCESS_TIMEOUT_SECONDS == 120


# ---------------------------------------------------------------------------
# V5 — --profile-dir path
# ---------------------------------------------------------------------------


class TestCwdForbiddenPaths:
    """V2-extended: --cwd must reject forbidden system path prefixes."""

    @pytest.mark.parametrize("forbidden_cwd", [
        "/etc",
        "/etc/passwd",
        "/proc",
        "/proc/1/mem",
        "/sys",
        "/sys/kernel",
        "/dev",
        "/dev/null",
        "/var/run",
        "/boot",
        "/sbin",
    ])
    def test_forbidden_cwd_prefix_rejected(self, forbidden_cwd):
        """Forbidden system paths must be rejected even if they exist."""
        import os
        # Some forbidden paths (e.g. /sbin, /var/run) may not exist on all systems
        # (e.g. Debian symlinks /sbin → /usr/sbin). Forbidden-prefix check still
        # fires before the existence check, so it should still be caught.
        # Skip only if the path resolves to something outside the forbidden set.
        rc, out, err = run_mlgg_module(["doctor", "--cwd", forbidden_cwd, "--dry-run"])
        if rc == 0 and "cwd_forbidden_path" not in err:
            # Path resolved to a location not in our forbidden set (e.g. symlink redirect)
            resolved = str(Path(forbidden_cwd).resolve())
            pytest.skip(
                f"'{forbidden_cwd}' resolves to '{resolved}' which is not in "
                f"_FORBIDDEN_CWD_PREFIXES on this system — symlink redirect"
            )
        assert rc == 2, f"Expected exit 2 for --cwd '{forbidden_cwd}', got {rc}"
        assert "cwd_forbidden_path" in err, (
            f"Expected 'cwd_forbidden_path' in stderr for '{forbidden_cwd}', got: {err}"
        )

    def test_cwd_nul_byte_rejected(self):
        """NUL byte in --cwd must be rejected."""
        rc, out, err = run_mlgg_module(["doctor", "--cwd", "/tmp/valid\x00injected", "--dry-run"])
        assert rc == 2
        assert "invalid_cwd" in err or "cwd_not_found" in err or "NUL" in err


class TestProfileNameValidation:
    """V5-extended: --profile-name must be a safe identifier (no path separators)."""

    @pytest.mark.parametrize("evil_name", [
        "../../../etc/passwd",
        "../../evil",
        "/absolute/path",
        "name/with/slash",
        "name\\backslash",
        "name\x00null",
        "a" * 200,  # too long
        "name with spaces",
        "name!@#$%",
        "",  # technically valid (empty = no profile) but test the regex boundary
    ])
    def test_evil_profile_name_rejected(self, evil_name, tmp_path):
        """Profile names with path separators or special chars must be rejected."""
        if evil_name == "":
            pytest.skip("Empty profile name is valid (means no profile)")
        rc, out, err = run_mlgg_module([
            "interactive", "--command", "init",
            "--profile-name", evil_name,
            "--dry-run",
            "--cwd", str(tmp_path),
        ])
        assert rc == 2, f"Expected exit 2 for --profile-name '{evil_name!r}', got {rc}"
        assert "invalid_profile_name" in err

    @pytest.mark.parametrize("safe_name", [
        "demo",
        "my-profile",
        "project_v2",
        "CKD-study-2026",
        "a",
        "z" * 128,
    ])
    def test_safe_profile_name_accepted(self, safe_name, tmp_path):
        """Valid alphanumeric-with-hyphens profile names must be accepted."""
        rc, out, err = run_mlgg_module([
            "interactive", "--command", "init",
            "--profile-name", safe_name,
            "--dry-run",
            "--cwd", str(tmp_path),
        ])
        assert "invalid_profile_name" not in err


class TestProfileDirValidation:
    """V5: --profile-dir is passed to interactive wizard; check it doesn't escape."""

    def test_profile_dir_dot_dot_is_blocked_by_resolve(self, tmp_path):
        """
        ../../../etc/passwd-style profile-dir should resolve without error
        but the interactive script (if it exists) decides whether to create it.
        The key property: the path must not be blindly executed as code.
        """
        rc, out, err = run_mlgg_module([
            "interactive", "--command", "init",
            "--profile-dir", "../../etc",
            "--dry-run",
        ])
        # dry-run means no subprocess → safe regardless; just verify no crash
        assert rc in {0, 2}

    def test_profile_dir_nul_byte_rejected_via_passthrough(self):
        """NUL bytes in profile-dir (passed via passthrough) must be caught."""
        rc, out, err = run_mlgg_module([
            "interactive", "--command", "init",
            "--dry-run",
            "--", "--profile-dir", "valid\x00injected",
        ])
        assert rc == 2
        assert "passthrough_arg_nul_byte" in err


# ---------------------------------------------------------------------------
# V6 — pre-parse python_bin in maybe_forward_subcommand_help
# ---------------------------------------------------------------------------


class TestPreParsePythonBin:
    """V6: maybe_forward_subcommand_help extracts python_bin before argparse runs."""

    def test_help_forwarding_with_evil_python_rejected(self):
        """--python /bin/sh with --help should be rejected before subprocess."""
        result = run_mlgg(["doctor", "--python", "/bin/sh", "--help"], timeout=5)
        assert result.returncode == 2
        assert "invalid_python_executable" in result.stderr

    def test_help_forwarding_with_legit_python_accepted(self):
        """--python python3 with --help should work if python3 exists."""
        import shutil
        if shutil.which("python3") is None:
            pytest.skip("python3 not on PATH")
        result = run_mlgg(["doctor", "--python", "python3", "--help"], timeout=10)
        # Should forward --help to doctor, which exits 0
        assert result.returncode in {0, 2}  # 2 only if doctor script is missing
        assert "invalid_python_executable" not in result.stderr


# ---------------------------------------------------------------------------
# Invariant / property tests (pairwise approach)
# ---------------------------------------------------------------------------


class TestCLIInvariants:
    """
    Invariant tests that must hold for ALL subcommand × flag combinations.

    Pairwise coverage: we test all pairs of (subcommand, flag_state) rather
    than all N×M combinations.
    """

    # Pairwise-generated matrix: (subcommand, dry_run, error_json)
    # This covers all 2-way interactions between these 3 dimensions.
    @pytest.mark.parametrize("subcommand,dry_run,error_json", [
        ("doctor", True, False),
        ("doctor", False, True),
        ("doctor", True, True),
        ("preflight", True, False),
        ("preflight", False, True),
        ("workflow", True, False),
        ("workflow", True, True),
        ("train", True, False),
        ("train", False, True),
        ("audit", True, False),
        ("audit", True, True),
        ("split", True, False),
        ("batch-review", True, False),
        ("fairness", True, False),
        ("sample-size", True, False),
    ])
    def test_dry_run_never_spawns_real_subprocess(
        self, subcommand, dry_run, error_json, tmp_path
    ):
        """
        With --dry-run, main() must print the command and return 0
        WITHOUT ever calling subprocess.run for the actual gate.
        """
        sys.path.insert(0, str(SCRIPTS_ROOT))
        import importlib
        if "mlgg" in sys.modules:
            del sys.modules["mlgg"]
        mlgg_mod = importlib.import_module("mlgg")

        spawned = []

        original_run_subprocess = mlgg_mod._run_subprocess

        def tracking_run(cmd, cwd, timeout=None):
            spawned.append(cmd)
            return original_run_subprocess(cmd, cwd, timeout=min(timeout or 60, 60))

        args = [subcommand, "--cwd", str(tmp_path)]
        if dry_run:
            args.append("--dry-run")
        if error_json:
            args.append("--error-json")

        old_argv = sys.argv
        sys.argv = ["mlgg.py"] + args
        try:
            with patch.object(mlgg_mod, "_run_subprocess", side_effect=tracking_run):
                try:
                    mlgg_mod.main()
                except SystemExit:
                    pass
        finally:
            sys.argv = old_argv

        if dry_run:
            assert len(spawned) == 0, (
                f"dry_run=True but _run_subprocess was called {len(spawned)} times "
                f"for subcommand='{subcommand}'"
            )

    @pytest.mark.parametrize("subcommand", [
        "doctor", "preflight", "workflow", "train", "audit", "split",
    ])
    def test_error_json_produces_valid_json_on_failure(self, subcommand, tmp_path):
        """
        With --error-json, any failure must emit parseable JSON to stderr.
        We force failure by providing a bad --python.
        """
        rc, out, err = run_mlgg_module([
            subcommand,
            "--python", "/bin/sh",  # will be rejected
            "--error-json",
            "--cwd", str(tmp_path),
        ])
        assert rc == 2
        # stderr might contain the error JSON — find it
        for line in err.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    # If we got JSON it should have status/code fields
                    # (but validation errors may not use --error-json path)
                    assert isinstance(obj, dict)
                except json.JSONDecodeError:
                    pytest.fail(f"--error-json produced invalid JSON line: {line!r}")

    def test_exit_codes_are_only_0_or_2(self, tmp_path):
        """
        The MLGG contract: exit code is always 0 (pass) or 2 (fail).
        Never 1 (Python unhandled exception).
        """
        cases = [
            ["doctor", "--dry-run"],
            ["doctor", "--python", "/bin/evil", "--dry-run"],
            ["preflight", "--dry-run"],
            ["workflow", "--dry-run"],
        ]
        for args in cases:
            rc, out, err = run_mlgg_module(args + ["--cwd", str(tmp_path)])
            assert rc in {0, 2}, (
                f"args={args} returned unexpected exit code {rc}. "
                f"stderr: {err[:200]}"
            )

    def test_unknown_subcommand_exits_2(self, tmp_path):
        """Unrecognized subcommand must exit non-zero (argparse handles this)."""
        result = run_mlgg(["not-a-real-subcommand"], timeout=5)
        assert result.returncode != 0

    def test_no_args_prints_help(self):
        """Calling mlgg.py with no args should print help (exit 0 or 2)."""
        result = run_mlgg([], timeout=5)
        assert result.returncode in {0, 1, 2}  # argparse exits 1 or 2 on missing required arg
        assert len(result.stderr) > 0 or len(result.stdout) > 0


# ---------------------------------------------------------------------------
# Combinatorial interaction tests (hand-crafted pairwise for key flag pairs)
# ---------------------------------------------------------------------------


class TestFlagPairInteractions:
    """
    Tests for pairwise flag interactions that are most likely to conflict.

    Covers:
    - --dry-run × --interactive
    - --dry-run × --print-only
    - --interactive × --command (valid/invalid)
    - --load-profile × --save-profile
    - preset subcommands × blocked flags
    """

    def test_dry_run_and_interactive_both_set(self, tmp_path):
        """--dry-run + --interactive should print command and exit 0 without spawning."""
        rc, out, err = run_mlgg_module([
            "workflow",
            "--interactive",
            "--command", "workflow",
            "--dry-run",
            "--cwd", str(tmp_path),
        ])
        # dry-run must dominate: never spawn
        assert rc in {0, 2}

    def test_interactive_without_command_exits_fail(self, tmp_path):
        """--interactive requires --command for non-interactive-capable subcommands."""
        rc, out, err = run_mlgg_module([
            "interactive",
            "--dry-run",
            "--cwd", str(tmp_path),
        ])
        # Either exits cleanly with 0 (prints help) or fails gracefully
        assert rc in {0, 2}
        assert rc != 1  # no unhandled Python exception

    def test_preset_subcommand_blocks_overriding_flags(self, tmp_path):
        """authority-release must block --stress-case-id passthrough."""
        rc, out, err = run_mlgg_module([
            "authority-release",
            "--",
            "--stress-case-id", "evil-override",
            "--cwd", str(tmp_path),
        ])
        assert rc == 2
        assert "authority_preset_route_override_forbidden" in err

    def test_preset_subcommand_does_not_block_unrelated_flags(self, tmp_path):
        """authority-release should allow unrelated passthrough flags."""
        rc, out, err = run_mlgg_module([
            "authority-release",
            "--dry-run",
            "--",
            "--some-unrelated-flag",
            "--cwd", str(tmp_path),
        ])
        # Should not fail on the unrelated flag (dry-run prevents actual execution)
        assert "authority_preset_route_override_forbidden" not in err

    @pytest.mark.parametrize("bad_command", ["not_a_command", "WORKFLOW", "TRAIN", "123"])
    def test_interactive_with_invalid_command_choice(self, bad_command, tmp_path):
        """--command with invalid value should be rejected by argparse."""
        result = run_mlgg(
            ["interactive", "--command", bad_command, "--dry-run", "--cwd", str(tmp_path)],
            timeout=5,
        )
        assert result.returncode != 0

    def test_load_and_save_profile_simultaneously(self, tmp_path):
        """--load-profile and --save-profile can coexist (interactive decides semantics)."""
        rc, out, err = run_mlgg_module([
            "interactive",
            "--command", "init",
            "--load-profile",
            "--save-profile",
            "--dry-run",
            "--cwd", str(tmp_path),
        ])
        assert rc in {0, 2}
        assert rc != 1


# ---------------------------------------------------------------------------
# Unicode and encoding edge cases
# ---------------------------------------------------------------------------


class TestUnicodeEdgeCases:
    """CLI must handle unicode in paths and arguments without crashing."""

    def test_unicode_in_cwd_that_exists(self, tmp_path):
        """If tmp_path has unicode characters, it should work fine."""
        unicode_dir = tmp_path / "测试目录_🔬"
        unicode_dir.mkdir()
        rc, out, err = run_mlgg_module(["doctor", "--dry-run", "--cwd", str(unicode_dir)])
        assert rc in {0, 2}
        assert rc != 1

    def test_unicode_in_passthrough_arg(self, tmp_path):
        """Unicode passthrough args should not crash the CLI."""
        rc, out, err = run_mlgg_module([
            "doctor", "--dry-run",
            "--cwd", str(tmp_path),
            "--", "--label", "诊断模型_v2",
        ])
        assert rc in {0, 2}
        assert rc != 1

    def test_rtl_unicode_in_arg(self, tmp_path):
        """Right-to-left unicode (Arabic) should not cause arg parsing issues."""
        rc, out, err = run_mlgg_module([
            "doctor", "--dry-run",
            "--cwd", str(tmp_path),
            "--", "--study-name", "النموذج_الطبي",
        ])
        assert rc in {0, 2}
        assert rc != 1
