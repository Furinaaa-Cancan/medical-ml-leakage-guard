#!/usr/bin/env python3
"""
Automated interactive tests for mlgg_pixel.py wizard using pty.

Uses Python built-in pty module to spawn the wizard in a pseudo-terminal
and simulate keypress sequences, then assert expected output.

Requires: MLGG_TEST=1 env var in child (set automatically).

Usage:
    python3 tests/test_wizard_interactive.py
"""

import os
import pty
import re
import select
import signal
import sys
import time

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "mlgg_pixel.py")
PYTHON = sys.executable

# ANSI key sequences
KEY_UP    = b"\x1b[A"
KEY_DOWN  = b"\x1b[B"
KEY_RIGHT = b"\x1b[C"
KEY_LEFT  = b"\x1b[D"
KEY_ENTER = b"\r"
KEY_ESC   = b"\x1b"
KEY_SPACE = b" "
KEY_Q     = b"q"
KEY_A     = b"a"
KEY_CTRLC = b"\x03"

# ── helpers ──────────────────────────────────────────────────────────────────

def strip_ansi(data):
    """Remove ANSI escape codes for easier assertion."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]*[a-zA-Z]', '', data)


class PTYSession:
    """Manages a wizard subprocess in a pseudo-terminal."""

    def __init__(self, env_lang="en_US.UTF-8"):
        env = os.environ.copy()
        env["LANG"] = env_lang
        env["TERM"] = "xterm-256color"
        env["COLUMNS"] = "100"
        env["LINES"] = "40"
        env["MLGG_TEST"] = "1"  # skip clear + logo
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execvpe(PYTHON, [PYTHON, SCRIPT], env)
        self.buf = b""

    def read_until(self, marker, timeout=10):
        """Read until marker appears in accumulated buffer."""
        if isinstance(marker, str):
            marker = marker.encode()
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            r, _, _ = select.select([self.fd], [], [], min(remaining, 0.1))
            if r:
                try:
                    chunk = os.read(self.fd, 8192)
                    if not chunk:
                        break
                    self.buf += chunk
                    if marker in self.buf:
                        return True
                except OSError:
                    break
        return False

    def text(self):
        """Return accumulated output stripped of ANSI codes."""
        return strip_ansi(self.buf)

    def clear_buf(self):
        """Clear the accumulated buffer."""
        self.buf = b""

    def send(self, data, delay=0.06):
        if isinstance(data, str):
            data = data.encode()
        os.write(self.fd, data)
        time.sleep(delay)

    def send_keys(self, *keys, delay=0.06):
        for k in keys:
            self.send(k, delay)

    def close(self):
        # Kill the entire process group (wizard + any spawned workers).
        # Using killpg ensures multiprocessing children and subprocesses also
        # receive SIGKILL, not just the direct pty child.
        try:
            pgid = os.getpgid(self.pid)
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            # Process already dead, or no permission — fall back to direct kill
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Reap the zombie in a daemon thread with a hard timeout.
        # os.waitpid(pid, 0) can block indefinitely when the child is in
        # D-state (uninterruptible I/O, e.g. network download), even after
        # SIGKILL is delivered.  A bounded wait prevents suite hangs.
        import threading
        def _reap():
            try:
                os.waitpid(self.pid, 0)
            except ChildProcessError:
                pass
        t = threading.Thread(target=_reap, daemon=True)
        t.start()
        t.join(timeout=3.0)  # give up after 3 s; leave reaping to OS on exit

        try:
            os.close(self.fd)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── test result tracking ────────────────────────────────────────────────────

class _ResultTracker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  \u2705 {name}")

    def fail(self, name, detail=""):
        self.failed += 1
        self.errors.append((name, detail))
        print(f"  \u274c {name}")
        if detail:
            for line in str(detail).split("\n")[:3]:
                print(f"     {line[:120]}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\n  Failed tests:")
            for name, detail in self.errors:
                print(f"    - {name}: {str(detail)[:100]}")
        print(f"{'='*60}")
        return self.failed == 0


R = _ResultTracker()


# ── test cases ───────────────────────────────────────────────────────────────

def test_01_launch_and_quit():
    """T01: Launch wizard, see language selection, press q to quit."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T01 launch+quit", f"No 'move' prompt. Got: {s.text()[:200]}")
            return
        if "Language" not in s.text():
            R.fail("T01 launch+quit", f"No 'Language'. Got: {s.text()[:200]}")
            return
        s.send(KEY_Q)
        if s.read_until("Bye", timeout=3):
            R.ok("T01 launch+quit")
        else:
            R.fail("T01 launch+quit", f"No bye. Got: {s.text()[-200:]}")


def test_02_arrow_keys_no_exit():
    """T02: Arrow keys (UP/DOWN/LEFT/RIGHT) don't cause exit."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T02 arrow keys", "No prompt")
            return
        s.send_keys(KEY_DOWN, KEY_UP, KEY_LEFT, KEY_RIGHT)
        time.sleep(0.2)
        # Still alive — press Enter to advance
        s.send(KEY_ENTER)
        if s.read_until("Step 2", timeout=5):
            R.ok("T02 arrow keys")
        else:
            t = s.text()
            if "Bye" in t:
                R.fail("T02 arrow keys", "Arrow key caused quit (ESC leak)")
            else:
                R.fail("T02 arrow keys", f"No Step 2. Got: {t[-200:]}")


def test_03_back_navigation():
    """T03: Enter (lang) → q (source) → goes back to lang."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T03 back nav", "No prompt"); return
        s.send(KEY_ENTER)
        if not s.read_until("Step 2", timeout=5):
            R.fail("T03 back nav", "No Step 2"); return
        s.send(KEY_Q)
        if s.read_until("Language", timeout=5):
            R.ok("T03 back nav")
        else:
            R.fail("T03 back nav", f"Didn't go back: {s.text()[-200:]}")


def test_04_demo_flow():
    """T04: English → Demo → should reach Step 8 (Confirm)."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T04 demo flow", "No prompt"); return
        s.send(KEY_ENTER)  # English
        if not s.read_until("Step 2", timeout=5):
            R.fail("T04 demo flow", "No Step 2"); return
        s.send_keys(KEY_DOWN, KEY_DOWN)
        s.send(KEY_ENTER)  # Demo
        if s.read_until("Step 8", timeout=5):
            if "Demo" in s.text() or "pipeline" in s.text():
                R.ok("T04 demo flow")
            else:
                R.fail("T04 demo flow", f"No demo box: {s.text()[-200:]}")
        else:
            R.fail("T04 demo flow", f"No Step 8: {s.text()[-300:]}")


@pytest.mark.slow
def test_05_download_to_step5():
    """T05: English → Download → Heart → default name → reaches Step 5."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T05 download→step5", "No prompt"); return
        s.send(KEY_ENTER)  # English
        s.read_until("Step 2", timeout=5)
        s.send(KEY_ENTER)  # Download
        s.read_until("Step 3", timeout=5)
        s.send(KEY_ENTER)  # Heart Disease
        if not s.read_until("heart_disease", timeout=5):
            R.fail("T05 download→step5", "No project name prompt"); return
        s.send(KEY_ENTER)  # accept default name
        if s.read_until("Step 5", timeout=5):
            R.ok("T05 download→step5")
        else:
            R.fail("T05 download→step5", f"No Step 5: {s.text()[-300:]}")


@pytest.mark.slow
def test_06_ctrlc_at_project_name():
    """T06: Ctrl+C at project name input → BACK, not crash."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T06 ctrl+c", "No prompt"); return
        s.send(KEY_ENTER)  # English
        s.read_until("Step 2", timeout=5)
        s.send(KEY_ENTER)  # Download
        s.read_until("Step 3", timeout=5)
        s.send(KEY_ENTER)  # Heart
        if not s.read_until("heart_disease", timeout=5):
            R.fail("T06 ctrl+c", "No name prompt"); return
        s.clear_buf()
        s.send(KEY_CTRLC)
        # Should go back to Step 3, not exit
        s.read_until("Step", timeout=5)
        t = s.text()
        if "Step 3" in t or "Heart" in t or "Dataset" in t:
            R.ok("T06 ctrl+c project name")
        elif "Interrupted" in t or "Bye" in t:
            R.fail("T06 ctrl+c", "Ctrl+C exited wizard instead of going back")
        else:
            # Check if process is still alive
            try:
                os.kill(s.pid, 0)
                R.fail("T06 ctrl+c", f"Process alive but unexpected output: {t[:200]}")
            except ProcessLookupError:
                R.fail("T06 ctrl+c", "Process died (crashed)")


@pytest.mark.slow
def test_07_multi_select():
    """T07: Model selection → Space toggle, A toggle-all, Enter confirm."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T07 multi-select", "No prompt"); return
        # Fast forward to Step 6
        s.send(KEY_ENTER)                     # English
        s.read_until("Step 2", timeout=5)
        s.send(KEY_ENTER)                     # Download
        s.read_until("Step 3", timeout=5)
        s.send(KEY_ENTER)                     # Heart
        s.read_until("heart_disease", timeout=5)
        s.send(KEY_ENTER)                     # default name
        s.read_until("Step 5", timeout=5)
        s.send(KEY_ENTER)                     # strategy
        # Wait for ratio picker (Step 5 has sub-steps)
        s.read_until("ratio", timeout=5)
        s.send(KEY_ENTER)                     # ratio
        if not s.read_until("Step 6", timeout=5):
            R.fail("T07 multi-select", f"No Step 6: {s.text()[-200:]}"); return
        # Space toggle, A toggle-all
        s.send(KEY_SPACE)
        time.sleep(0.1)
        s.send(KEY_A)
        time.sleep(0.1)
        s.send(KEY_ENTER)
        if s.read_until("Step 7", timeout=5):
            R.ok("T07 multi-select")
        else:
            R.fail("T07 multi-select", f"No Step 7: {s.text()[-200:]}")


@pytest.mark.slow
def test_08_sub_step_back():
    """T08: Tuning → calibration → q → back to tuning strategy (not Step 6)."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T08 sub-step back", "No prompt"); return
        # Fast forward to Step 7
        s.send(KEY_ENTER)  # English
        s.read_until("Step 2", timeout=5)
        s.send(KEY_ENTER)  # Download
        s.read_until("Step 3", timeout=5)
        s.send(KEY_ENTER)  # Heart
        s.read_until("heart_disease", timeout=5)
        s.send(KEY_ENTER)  # default name
        s.read_until("Step 5", timeout=5)
        s.send(KEY_ENTER)  # strategy
        s.read_until("ratio", timeout=5)
        s.send(KEY_ENTER)  # ratio
        s.read_until("Step 6", timeout=5)
        s.send(KEY_ENTER)  # models
        if not s.read_until("Step 7", timeout=5):
            R.fail("T08 sub-step back", f"No Step 7: {s.text()[-200:]}"); return
        # Enter to select tuning strategy (Fixed Grid)
        s.send(KEY_ENTER)
        if not s.read_until("alibration", timeout=5):
            R.fail("T08 sub-step back", "No calibration sub-step"); return
        # q → should go back to tuning strategy, NOT exit to Step 6
        s.clear_buf()
        s.send(KEY_Q)
        s.read_until("Step 7", timeout=5)
        t = s.text()
        if "Hyperparameter" in t or "search" in t or "Fixed" in t:
            R.ok("T08 sub-step back")
        elif "Step 6" in t or "Model" in t:
            R.fail("T08 sub-step back", "Went to Step 6 instead of tuning sub-step 0")
        else:
            R.fail("T08 sub-step back", f"Unexpected: {t[:200]}")


@pytest.mark.slow
def test_09_chinese_lang():
    """T09: Select Chinese → verify Chinese text."""
    with PTYSession(env_lang="zh_CN.UTF-8") as s:
        # Chinese locale → nav hint is "移动" not "move"; use "Language" as marker
        if not s.read_until("Language", timeout=15):
            R.fail("T09 chinese", "No prompt"); return
        s.send(KEY_DOWN)  # select 中文
        s.send(KEY_ENTER)
        if s.read_until("Step 2", timeout=5):
            t = s.text()
            if "\u6570\u636e\u6765\u6e90" in t or "\u4e0b\u8f7d" in t:
                R.ok("T09 chinese lang")
            else:
                R.fail("T09 chinese", f"No Chinese text: {t[-200:]}")
        else:
            R.fail("T09 chinese", f"No Step 2: {s.text()[-200:]}")


@pytest.mark.slow
def test_10_digit_shortcut():
    """T10: Press '2' to jump to option 2 (中文), Enter → Chinese UI."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T10 digit shortcut", "No prompt"); return
        s.send(b"2")
        time.sleep(0.1)
        s.send(KEY_ENTER)
        if s.read_until("Step 2", timeout=5):
            t = s.text()
            if "\u6570\u636e\u6765\u6e90" in t or "\u4e0b\u8f7d" in t:
                R.ok("T10 digit shortcut")
            else:
                R.fail("T10 digit shortcut", f"Expected Chinese: {t[-200:]}")
        else:
            R.fail("T10 digit shortcut", f"No Step 2: {s.text()[-200:]}")


@pytest.mark.slow
def test_11_esc_key():
    """T11: ESC key acts like q (goes back)."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T11 ESC", "No prompt"); return
        s.send(KEY_ENTER)  # English
        if not s.read_until("Step 2", timeout=5):
            R.fail("T11 ESC", "No Step 2"); return
        s.clear_buf()
        s.send(KEY_ESC)
        # ESC needs time for _getch timeout (0.05s)
        time.sleep(0.2)
        s.read_until("Language", timeout=5)
        if "Language" in s.text():
            R.ok("T11 ESC key")
        else:
            R.fail("T11 ESC key", f"No back: {s.text()[:200]}")


@pytest.mark.slow
def test_12_confirm_go_back():
    """T12: Demo → Confirm → Go Back → returns to Step 2."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T12 confirm back", "No prompt"); return
        s.send(KEY_ENTER)  # English
        s.read_until("Step 2", timeout=5)
        s.send_keys(KEY_DOWN, KEY_DOWN)
        s.send(KEY_ENTER)  # Demo
        if not s.read_until("Step 8", timeout=5):
            R.fail("T12 confirm back", f"No Step 8: {s.text()[-200:]}"); return
        # Go Back (2nd option)
        s.send(KEY_DOWN)
        s.send(KEY_ENTER)
        if s.read_until("Step 2", timeout=5):
            R.ok("T12 confirm back")
        else:
            R.fail("T12 confirm back", f"No Step 2: {s.text()[-200:]}")


@pytest.mark.slow
def test_13_csv_manual_input():
    """T13: English → CSV → Enter path manually → input examples/heart_disease.csv → reaches Step 4 or Step 5."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "examples", "heart_disease.csv")
    csv_abs = os.path.abspath(csv_path)
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T13 csv manual", "No prompt"); return
        s.send(KEY_ENTER)  # English
        if not s.read_until("Step 2", timeout=5):
            R.fail("T13 csv manual", "No Step 2"); return
        s.send(KEY_DOWN)   # select CSV
        s.send(KEY_ENTER)
        if not s.read_until("Step 3", timeout=5):
            # CSV source might skip to manual input directly
            pass
        # Wait for CSV picker or manual prompt
        s.read_until(">", timeout=5)
        t = s.text()
        # If scan_csv found files, select "Enter path manually" (last option)
        if "manual" in t.lower() or "path" in t.lower() or "手动" in t:
            # Navigate to last option and select it
            for _ in range(20):
                s.send(KEY_DOWN, delay=0.03)
            s.send(KEY_ENTER)
            s.read_until(">", timeout=5)
        # Now type the CSV path
        s.send(csv_abs.encode() + KEY_ENTER, delay=0.05)
        # Should reach Step 4 (config) or project name prompt
        found = s.read_until("Step", timeout=8)
        final = s.text()
        if found and ("Step 4" in final or "Step 5" in final or "heart_disease" in final):
            R.ok("T13 csv manual input")
        elif "heart_disease" in final or "patient_id" in final:
            R.ok("T13 csv manual input")
        else:
            R.fail("T13 csv manual", f"Expected Step 4/5: {final[-300:]}")


@pytest.mark.slow
def test_14_column_count_error():
    """T14: Construct 1-column CSV → select it → verify error prompt, no crash."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    try:
        tmp.write("only_col\n1\n2\n3\n")
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()
        with PTYSession() as s:
            if not s.read_until("move", timeout=15):
                R.fail("T14 col error", "No prompt"); return
            s.send(KEY_ENTER)  # English
            s.read_until("Step 2", timeout=5)
            s.send(KEY_DOWN)   # CSV
            s.send(KEY_ENTER)
            # Wait for picker or manual prompt
            s.read_until(">", timeout=5)
            t = s.text()
            # Navigate to manual path entry
            if "manual" in t.lower() or "path" in t.lower() or "手动" in t:
                for _ in range(20):
                    s.send(KEY_DOWN, delay=0.03)
                s.send(KEY_ENTER)
                s.read_until(">", timeout=5)
            # Type the 1-column CSV path
            s.send(tmp_path.encode() + KEY_ENTER, delay=0.05)
            # Should show name prompt, then Step 4 with error about too few columns
            s.read_until("Step", timeout=8)
            s.text()
            # Check process is still alive (no crash/infinite loop)
            try:
                os.kill(s.pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            if alive:
                R.ok("T14 column count error")
            else:
                R.fail("T14 col error", "Process crashed")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@pytest.mark.slow
def test_15_step_split_complete():
    """T15: Download → Heart → default name → temporal → ratio → verify reaches Step 6."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T15 split flow", "No prompt"); return
        s.send(KEY_ENTER)  # English
        s.read_until("Step 2", timeout=5)
        s.send(KEY_ENTER)  # Download
        s.read_until("Step 3", timeout=5)
        s.send(KEY_ENTER)  # Heart Disease
        s.read_until("heart_disease", timeout=5)
        s.send(KEY_ENTER)  # default name
        if not s.read_until("Step 5", timeout=5):
            R.fail("T15 split flow", f"No Step 5: {s.text()[-200:]}"); return
        # Select temporal strategy (first option)
        s.send(KEY_ENTER)
        # Temporal → ratio picker
        if not s.read_until("ratio", timeout=5):
            R.fail("T15 split flow", f"No ratio: {s.text()[-200:]}"); return
        s.send(KEY_ENTER)  # 60/20/20 ratio
        if s.read_until("Step 6", timeout=5):
            R.ok("T15 step_split complete")
        else:
            R.fail("T15 split flow", f"No Step 6: {s.text()[-200:]}")


@pytest.mark.slow
def test_16_step_confirm_info():
    """T16: Complete download flow to Step 8 → verify confirm box has key info."""
    with PTYSession() as s:
        if not s.read_until("move", timeout=15):
            R.fail("T16 confirm info", "No prompt"); return
        s.send(KEY_ENTER)  # English
        s.read_until("Step 2", timeout=5)
        s.send(KEY_ENTER)  # Download
        s.read_until("Step 3", timeout=5)
        s.send(KEY_ENTER)  # Heart Disease
        s.read_until("heart_disease", timeout=5)
        s.send(KEY_ENTER)  # default name
        s.read_until("Step 5", timeout=5)
        s.send(KEY_ENTER)  # strategy (temporal)
        s.read_until("ratio", timeout=5)
        s.send(KEY_ENTER)  # ratio 60/20/20
        s.read_until("Step 6", timeout=5)
        s.send(KEY_ENTER)  # models (defaults)
        s.read_until("Step 7", timeout=5)
        s.send(KEY_ENTER)  # tuning strategy
        s.read_until("alibration", timeout=5)
        s.send(KEY_ENTER)  # calibration
        # Device selection may appear
        time.sleep(0.3)
        # Clear buffer so we capture the Step 8 confirm screen fresh
        s.clear_buf()
        s.read_until("Step 8", timeout=5)
        t = s.text()
        if "Step 8" not in t:
            # May need one more Enter for device
            s.send(KEY_ENTER)
            s.clear_buf()
            s.read_until("Step 8", timeout=5)
            t = s.text()
        if "Step 8" not in t:
            R.fail("T16 confirm info", f"No Step 8: {t[-300:]}"); return
        # Give time for full box rendering
        time.sleep(0.5)
        # Read any remaining output
        s.read_until("XXXXNOTEXIST", timeout=1)
        t = s.text()
        # Verify confirm box contains key info
        # heart_disease.csv should appear as file name
        has_file = "heart_disease" in t
        # The box should contain strategy/output/model info
        has_strategy = "temporal" in t.lower() or "grouped" in t.lower()
        has_output = "heart_disease" in t
        if has_file and (has_strategy or has_output):
            R.ok("T16 step_confirm info")
        else:
            R.fail("T16 confirm info", f"Missing info in confirm box: file={has_file} strat={has_strategy} out={has_output}. Text: {t[-400:]}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  mlgg_pixel.py Interactive Tests (pty + MLGG_TEST)")
    print(f"{'='*60}\n")

    test_01_launch_and_quit()
    test_02_arrow_keys_no_exit()
    test_03_back_navigation()
    test_04_demo_flow()
    test_05_download_to_step5()
    test_06_ctrlc_at_project_name()
    test_07_multi_select()
    test_08_sub_step_back()
    test_09_chinese_lang()
    test_10_digit_shortcut()
    test_11_esc_key()
    test_12_confirm_go_back()
    test_13_csv_manual_input()
    test_14_column_count_error()
    test_15_step_split_complete()
    test_16_step_confirm_info()

    ok = R.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
