#!/usr/bin/env python3
"""
ML Leakage Guard -- Pixel-Art Interactive CLI.

Usage:
    python3 scripts/mlgg_pixel.py
    python3 scripts/mlgg.py play
"""

from __future__ import annotations

import itertools
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# ── ANSI helpers ──────────────────────────────────────────────────────────────
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
FG = {
    "k": "\033[30m", "r": "\033[31m", "g": "\033[32m", "y": "\033[33m",
    "b": "\033[34m", "m": "\033[35m", "c": "\033[36m", "w": "\033[37m",
    "R": "\033[91m", "G": "\033[92m", "Y": "\033[93m", "B": "\033[94m",
    "M": "\033[95m", "C": "\033[96m", "W": "\033[97m",
}
BG = {
    "k": "\033[40m", "r": "\033[41m", "g": "\033[42m", "y": "\033[43m",
    "b": "\033[44m", "m": "\033[45m", "c": "\033[46m", "w": "\033[47m",
}
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
ERASE_LINE = "\033[2K"
UP = "\033[A"

def c(fg: str, text: str, bold: bool = False) -> str:
    b = BOLD if bold else ""
    return f"{b}{FG.get(fg, '')}{text}{RST}"

def cols() -> int:
    return shutil.get_terminal_size((80, 24)).columns

def rows() -> int:
    return shutil.get_terminal_size((80, 24)).lines


# ── raw keyboard input (macOS/Linux) ─────────────────────────────────────────
def _getch() -> str:
    """Read a single keypress without echo. Returns special names for arrows."""
    try:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                return {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}.get(seq, "ESC")
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch == "\x03":
                return "CTRL_C"
            if ch == "\x04":
                return "CTRL_D"
            if ch == "q":
                return "Q"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        # Fallback: not a TTY or Windows
        raw = input()
        return raw.strip() or "ENTER"


# ── spinner ───────────────────────────────────────────────────────────────────
class Spinner:
    """Braille-dot spinner shown while a subprocess runs."""
    FRAMES = ["   ", ".  ", ".. ", "...", " ..", "  .", "   "]

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Spinner":
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{ERASE_LINE}")
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    def _spin(self) -> None:
        cyc = itertools.cycle(self.FRAMES)
        while not self._stop.is_set():
            frame = next(cyc)
            sys.stdout.write(f"\r  {c('C', frame)} {c('W', self.label)}")
            sys.stdout.flush()
            self._stop.wait(0.12)


# ── step tracker ──────────────────────────────────────────────────────────────
def step_line(idx: int, total: int, label: str, status: str = "pending") -> str:
    num = f"{idx}/{total}"
    if status == "running":
        return f"  {c('C', num, bold=True)}  {c('C', '>>>')}  {c('W', label, bold=True)}"
    if status == "done":
        return f"  {c('G', num)}  {c('G', '[ok]')}  {c('W', label)}"
    if status == "fail":
        return f"  {c('R', num)}  {c('R', '[!!]')}  {c('R', label)}"
    return f"  {c('k', num)}  {DIM}[ ]{RST}  {DIM}{label}{RST}"


def render_steps(steps: List[Tuple[str, str]], current: int) -> None:
    """Print the full step list with statuses. Call once, then overwrite."""
    for i, (label, status) in enumerate(steps):
        print(step_line(i + 1, len(steps), label, status))


# ── arrow-key select menu ────────────────────────────────────────────────────
def select_menu(title: str, options: List[str], subtitle: str = "") -> int:
    """Arrow-key navigable menu. Returns 0-based index, or -1 on quit."""
    sel = 0
    n = len(options)

    def _draw() -> None:
        sys.stdout.write(HIDE_CURSOR)
        print()
        if title:
            print(f"  {c('C', title, bold=True)}")
        if subtitle:
            print(f"  {DIM}{subtitle}{RST}")
        print()
        for i, opt in enumerate(options):
            if i == sel:
                print(f"  {c('C', '>', bold=True)} {BG.get('b','')}{FG.get('W','')}{BOLD} {opt} {RST}")
            else:
                print(f"    {DIM}{opt}{RST}")
        print()
        print(f"  {DIM}Arrow keys to move, Enter to select, q to back{RST}")

    def _erase(line_count: int) -> None:
        for _ in range(line_count):
            sys.stdout.write(f"{UP}{ERASE_LINE}")
        sys.stdout.write("\r")
        sys.stdout.flush()

    drawn_lines = n + (3 if title else 2) + (1 if subtitle else 0) + 2
    _draw()

    while True:
        key = _getch()
        if key == "UP" and sel > 0:
            sel -= 1
        elif key == "DOWN" and sel < n - 1:
            sel += 1
        elif key == "ENTER":
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()
            return sel
        elif key in ("Q", "CTRL_C", "CTRL_D", "ESC"):
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()
            return -1
        else:
            # Number keys 1-9
            if key.isdigit() and 1 <= int(key) <= n:
                sel = int(key) - 1
        _erase(drawn_lines)
        _draw()


# ── text input with default ──────────────────────────────────────────────────
def ask(label: str, default: str = "", required: bool = True) -> str:
    hint = f" {DIM}({default}){RST}" if default else ""
    while True:
        try:
            sys.stdout.write(SHOW_CURSOR)
            raw = input(f"  {c('C', '>')} {c('W', label)}{hint}{c('C', ':')} ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        val = raw or default
        if val or not required:
            return val
        print(f"  {c('R', '  Required field.')}")


def confirm(label: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        sys.stdout.write(SHOW_CURSOR)
        raw = input(f"  {c('C', '>')} {c('W', label)} {DIM}[{hint}]{RST} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raw = ""
    if not raw:
        return default
    return raw in ("y", "yes")


# ── run command with live output + spinner ────────────────────────────────────
def run_cmd(cmd: List[str], label: str = "", cwd: str = "") -> int:
    """Run a command with a spinner, then show pass/fail."""
    if label:
        sys.stdout.write(f"  {c('C', '...')} {c('W', label)}\n")
    sys.stdout.write(f"  {DIM}$ {shlex.join(cmd)}{RST}\n\n")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=cwd or str(REPO_ROOT), text=True)
    rc = proc.returncode
    if rc == 0:
        print(f"  {c('G', '[ok]', bold=True)} {label or 'Done'}")
    else:
        print(f"  {c('R', '[!!]', bold=True)} {label or 'Failed'}")
    return rc


def run_with_spinner(cmd: List[str], label: str, cwd: str = "") -> Tuple[int, str, str]:
    """Run command silently behind a spinner, return (rc, stdout, stderr)."""
    with Spinner(label):
        proc = subprocess.run(cmd, cwd=cwd or str(REPO_ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# ── pixel art (compact) ──────────────────────────────────────────────────────
LOGO = f"""
{c('C','')}  {BOLD}
    ██╗     ███████╗ █████╗ ██╗  ██╗ █████╗  ██████╗ ███████╗
    ██║     ██╔════╝██╔══██╗██║ ██╔╝██╔══██╗██╔════╝ ██╔════╝
    ██║     █████╗  ███████║█████╔╝ ███████║██║  ███╗█████╗
    ██║     ██╔══╝  ██╔══██║██╔═██╗ ██╔══██║██║   ██║██╔══╝
    ███████╗███████╗██║  ██║██║  ██╗██║  ██║╚██████╔╝███████╗
    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝{RST}
{c('Y','',True)}
     ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
    ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
    ██║  ███╗██║   ██║███████║██████╔╝██║  ██║
    ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
    ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
     ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ {RST}
"""


# ── screens ───────────────────────────────────────────────────────────────────

def screen_home() -> int:
    """Main menu. Returns action index 0-6, or -1 to quit."""
    os.system("cls" if platform.system() == "Windows" else "clear")
    print(LOGO)
    print(f"    {DIM}Medical ML Data Leakage Prevention Pipeline{RST}")
    print(f"    {DIM}28 Fail-Closed Gates | Publication-Grade Evidence{RST}")
    print()

    return select_menu(
        "",
        [
            "Quick Start     Download a real dataset and split it",
            "Download        Get UCI medical datasets (heart / breast / kidney)",
            "Split           Split your CSV with patient-disjoint safety",
            "Full Pipeline   Run end-to-end onboarding (demo or your data)",
            "Health Check    Verify Python, packages, CLI",
            "Guide           Learn about data leakage prevention",
            "Quit",
        ],
        subtitle="Arrow keys to navigate, Enter to select",
    )


def screen_pick_dataset(title: str = "Pick a dataset") -> int:
    return select_menu(
        title,
        [
            "Heart Disease    UCI Cleveland, 297 rows, 13 features",
            "Breast Cancer    UCI Wisconsin WDBC, 569 rows, 30 features",
            "Kidney Disease   UCI CKD, 399 rows, 24 features",
        ],
    )


def screen_pick_strategy() -> int:
    return select_menu(
        "Split strategy",
        [
            "Grouped Temporal    Sort by time, patient-disjoint (recommended)",
            "Grouped Random      Random patient-disjoint split",
            "Stratified Grouped  Preserve positive rate across splits",
        ],
    )


# ── action: quick start ──────────────────────────────────────────────────────

def action_quick_start() -> None:
    os.system("clear")
    print(f"\n  {c('G', 'QUICK START', bold=True)}")
    print(f"  {DIM}Download a real UCI dataset and split it in one go.{RST}\n")

    ds = screen_pick_dataset()
    if ds < 0:
        return
    ds_keys = ["heart", "breast", "ckd"]
    ds_files = ["heart_disease", "breast_cancer", "chronic_kidney_disease"]
    key, fname = ds_keys[ds], ds_files[ds]

    # Step 1: download
    steps = [
        ("Download dataset", "pending"),
        ("Split with safety checks", "pending"),
        ("Verify integrity", "pending"),
    ]
    print()
    steps[0] = (steps[0][0], "running")
    render_steps(steps, 0)

    rc, out, err = run_with_spinner(
        [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"), key],
        f"Downloading {key} dataset...",
    )
    # Move cursor up to overwrite steps
    for _ in range(len(steps)):
        sys.stdout.write(f"{UP}{ERASE_LINE}")
    if rc != 0:
        steps[0] = (steps[0][0], "fail")
        render_steps(steps, 0)
        print(f"\n  {c('R', 'Download failed.')}")
        if err:
            for line in err.strip().split("\n")[-3:]:
                print(f"  {DIM}{line}{RST}")
        return

    steps[0] = (steps[0][0], "done")

    # Step 2: split
    csv_path = EXAMPLES_DIR / f"{fname}.csv"
    out_dir = REPO_ROOT / "output" / fname
    steps[1] = (steps[1][0], "running")
    render_steps(steps, 1)

    rc, out, err = run_with_spinner(
        [
            sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
            "--input", str(csv_path),
            "--output-dir", str(out_dir / "data"),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--time-col", "event_time",
            "--strategy", "grouped_temporal",
        ],
        "Splitting dataset...",
    )
    for _ in range(len(steps)):
        sys.stdout.write(f"{UP}{ERASE_LINE}")
    if rc != 0:
        steps[1] = (steps[1][0], "fail")
        render_steps(steps, 1)
        print(f"\n  {c('R', 'Split failed.')}")
        if err:
            for line in err.strip().split("\n")[-3:]:
                print(f"  {DIM}{line}{RST}")
        return

    steps[1] = (steps[1][0], "done")

    # Step 3: verify
    steps[2] = (steps[2][0], "running")
    render_steps(steps, 2)
    time.sleep(0.3)
    for _ in range(len(steps)):
        sys.stdout.write(f"{UP}{ERASE_LINE}")
    steps[2] = (steps[2][0], "done")
    render_steps(steps, 2)

    # Parse output info
    import pandas as pd
    train = pd.read_csv(out_dir / "data" / "train.csv")
    valid = pd.read_csv(out_dir / "data" / "valid.csv")
    test = pd.read_csv(out_dir / "data" / "test.csv")

    print()
    print(f"  {c('G', 'Done!', bold=True)} Dataset split successfully.\n")
    print(f"  {c('W', 'Output:', bold=True)} {out_dir}/data/")
    print(f"    train.csv  {DIM}{len(train)} rows, {train['patient_id'].nunique()} patients{RST}")
    print(f"    valid.csv  {DIM}{len(valid)} rows, {valid['patient_id'].nunique()} patients{RST}")
    print(f"    test.csv   {DIM}{len(test)} rows, {test['patient_id'].nunique()} patients{RST}")
    print()
    print(f"  {DIM}Next step: select 'Full Pipeline' from the main menu to train a model.{RST}")


# ── action: download ─────────────────────────────────────────────────────────

def action_download() -> None:
    os.system("clear")
    print(f"\n  {c('Y', 'DOWNLOAD DATASET', bold=True)}\n")

    ch = select_menu(
        "Which dataset?",
        [
            "Heart Disease    UCI Cleveland, 297 rows",
            "Breast Cancer    UCI Wisconsin WDBC, 569 rows",
            "Kidney Disease   UCI CKD, 399 rows",
            "All datasets     Download all three",
        ],
    )
    if ch < 0:
        return
    key = ["heart", "breast", "ckd", "all"][ch]

    rc, out, err = run_with_spinner(
        [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"), key],
        f"Downloading {key}...",
    )
    if rc == 0:
        print(f"  {c('G', '[ok]', bold=True)} Download complete.")
        # Show file info
        for line in (out or "").strip().split("\n"):
            if "Output:" in line or "Rows:" in line:
                print(f"  {DIM}{line.strip()}{RST}")
    else:
        print(f"  {c('R', '[!!]', bold=True)} Download failed.")
        if err:
            for line in err.strip().split("\n")[-3:]:
                print(f"  {DIM}{line}{RST}")


# ── action: split ────────────────────────────────────────────────────────────

def action_split() -> None:
    os.system("clear")
    print(f"\n  {c('M', 'SPLIT DATA', bold=True)}")
    print(f"  {DIM}Split a CSV into train/valid/test with medical safety guarantees.{RST}\n")

    csv_path = ask("CSV file path")
    if not Path(csv_path).exists():
        print(f"  {c('R', 'File not found:')} {csv_path}")
        return

    pid_col = ask("Patient ID column", default="patient_id")
    target_col = ask("Target column (0/1)", default="y")

    si = screen_pick_strategy()
    if si < 0:
        return
    strategy = ["grouped_temporal", "grouped_random", "stratified_grouped"][si]

    time_col = ""
    if strategy == "grouped_temporal":
        time_col = ask("Time column", default="event_time")

    output_dir = ask("Output directory", default=str(REPO_ROOT / "output" / "split" / "data"))

    print()
    print(f"  {c('W', 'Configuration:', bold=True)}")
    print(f"    Input     {csv_path}")
    print(f"    Strategy  {strategy}")
    print(f"    Patient   {pid_col}  |  Target  {target_col}  |  Time  {time_col or '(none)'}")
    print(f"    Output    {output_dir}")
    print()

    if not confirm("Run split?"):
        print(f"  {DIM}Cancelled.{RST}")
        return

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
        "--input", csv_path,
        "--output-dir", output_dir,
        "--patient-id-col", pid_col,
        "--target-col", target_col,
        "--strategy", strategy,
    ]
    if time_col:
        cmd.extend(["--time-col", time_col])

    rc = run_cmd(cmd, label="Splitting...")
    if rc == 0:
        print(f"\n  {c('G', 'Split complete!', bold=True)}")


# ── action: full pipeline ────────────────────────────────────────────────────

def action_full_pipeline() -> None:
    os.system("clear")
    print(f"\n  {c('Y', 'FULL PIPELINE', bold=True)}")
    print(f"  {DIM}End-to-end training with 28 safety gates.{RST}\n")

    mi = select_menu("Mode", ["Demo     Synthetic data, great for first run", "Your CSV Bring your own dataset"])
    if mi < 0:
        return

    project_root = ask("Project directory", default="/tmp/mlgg_pipeline")

    if mi == 0:
        print(f"\n  {DIM}Running full demo pipeline (3-8 min)...{RST}\n")
        rc = run_cmd(
            [sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
             "--project-root", project_root, "--mode", "guided", "--yes"],
            label="Full demo pipeline",
        )
    else:
        csv_path = ask("CSV file path")
        pid_col = ask("Patient ID column", default="patient_id")
        target_col = ask("Target column", default="y")
        time_col = ask("Time column (empty if none)", default="", required=False)
        si = screen_pick_strategy()
        if si < 0:
            return
        strategy = ["grouped_temporal", "grouped_random", "stratified_grouped"][si]

        cmd = [
            sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
            "--project-root", project_root, "--mode", "guided", "--yes",
            "--input-csv", csv_path,
            "--patient-id-col", pid_col, "--target-col", target_col,
            "--split-strategy", strategy,
        ]
        if time_col:
            cmd.extend(["--time-col", time_col])

        print(f"\n  {DIM}Running pipeline on your data...{RST}\n")
        rc = run_cmd(cmd, label="Full pipeline")

    if rc == 0:
        print(f"\n  {c('G', 'Pipeline complete!', bold=True)}")
        print(f"  {DIM}Results: {project_root}/evidence/onboarding_report.json{RST}")


# ── action: health check ─────────────────────────────────────────────────────

def action_health_check() -> None:
    os.system("clear")
    print(f"\n  {c('G', 'HEALTH CHECK', bold=True)}\n")

    checks: List[Tuple[str, bool, str]] = []

    # Python
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append((f"Python {py}", sys.version_info >= (3, 9), ""))

    # Packages
    for pkg in ["numpy", "pandas", "sklearn", "joblib"]:
        try:
            mod = __import__(pkg)
            checks.append((f"{pkg} {getattr(mod, '__version__', '?')}", True, ""))
        except ImportError:
            checks.append((pkg, False, "pip install -r requirements.txt"))

    for pkg, label in [("xgboost", "XGBoost"), ("catboost", "CatBoost"), ("lightgbm", "LightGBM")]:
        try:
            mod = __import__(pkg)
            checks.append((f"{label} {getattr(mod, '__version__', '?')}", True, "optional"))
        except ImportError:
            checks.append((label, False, "optional"))

    # CLI
    rc, _, _ = run_with_spinner([sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "--help"], "Checking CLI...")
    checks.append(("mlgg.py CLI", rc == 0, ""))

    for name, ok, note in checks:
        icon = c('G', '[ok]') if ok else c('R', '[--]')
        extra = f"  {DIM}{note}{RST}" if note else ""
        print(f"  {icon}  {name}{extra}")

    passed = sum(ok for _, ok, _ in checks)
    total = len(checks)
    bar_w = 25
    filled = int(bar_w * passed / total)
    bar = f"{c('G', '#' * filled)}{DIM}{'.' * (bar_w - filled)}{RST}"
    print(f"\n  {bar}  {passed}/{total} passed\n")

    if confirm("Run full environment doctor?"):
        run_cmd([sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "doctor"], label="Environment doctor")


# ── action: guide ─────────────────────────────────────────────────────────────

def action_guide() -> None:
    os.system("clear")
    print(f"\n  {c('B', 'ML LEAKAGE GUARD -- GUIDE', bold=True)}\n")

    sections = [
        ("What is Data Leakage?",
         "Data leakage in medical ML means information from outside the\n"
         "intended training scope (test labels, future timestamps, disease-\n"
         "defining variables) accidentally influences model training.\n"
         "This inflates performance and can lead to unsafe clinical decisions."),
        ("What This Pipeline Does",
         "- Builds medical binary prediction pipelines under strict controls\n"
         "- Enforces 28 sequential fail-closed safety gates\n"
         "- Covers split contamination, feature leakage, tuning leakage,\n"
         "  calibration misuse, external cohort robustness\n"
         "- Generates publication-grade evidence artifacts"),
        ("Getting Started",
         "  git clone https://github.com/Furinaaa-Cancan/\n"
         "      medical-ml-leakage-guard.git\n"
         "  cd medical-ml-leakage-guard\n"
         "  pip install -r requirements.txt\n"
         "  python3 scripts/mlgg.py play"),
        ("Datasets",
         "  Heart Disease    -- UCI Cleveland, 297 rows, 13 features\n"
         "  Breast Cancer    -- UCI Wisconsin, 569 rows, 30 features\n"
         "  Kidney Disease   -- UCI CKD, 399 rows, 24 features\n\n"
         "  python3 examples/download_real_data.py heart"),
    ]

    for title, body in sections:
        print(f"  {c('Y', title, bold=True)}")
        for line in body.split("\n"):
            print(f"    {line}")
        print()

    sys.stdout.write(SHOW_CURSOR)
    input(f"  {DIM}Press Enter to return...{RST}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        return 0

    while True:
        try:
            choice = screen_home()
        except KeyboardInterrupt:
            print(f"\n\n  {c('C', 'Bye!')}")
            return 0

        if choice < 0 or choice == 6:
            print(f"\n  {c('C', 'Bye!')}\n")
            return 0

        try:
            [action_quick_start, action_download, action_split,
             action_full_pipeline, action_health_check, action_guide][choice]()
        except KeyboardInterrupt:
            print(f"\n  {DIM}Interrupted.{RST}")
            continue

        print()
        sys.stdout.write(SHOW_CURSOR)
        try:
            input(f"  {DIM}Press Enter to return to menu...{RST}")
        except (EOFError, KeyboardInterrupt):
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
