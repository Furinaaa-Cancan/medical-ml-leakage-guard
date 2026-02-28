#!/usr/bin/env python3
"""
ML Leakage Guard -- Pixel-Art Interactive CLI.

Usage:
    python3 scripts/mlgg_pixel.py
    python3 scripts/mlgg.py play
"""

from __future__ import annotations

import csv
import glob
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
DESKTOP = Path.home() / "Desktop"
DEFAULT_OUT = DESKTOP / "MLGG_Output"

# ── ANSI ──────────────────────────────────────────────────────────────────────
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
FG = {
    "k": "\033[30m", "r": "\033[31m", "g": "\033[32m", "y": "\033[33m",
    "b": "\033[34m", "m": "\033[35m", "c": "\033[36m", "w": "\033[37m",
    "R": "\033[91m", "G": "\033[92m", "Y": "\033[93m", "B": "\033[94m",
    "M": "\033[95m", "C": "\033[96m", "W": "\033[97m",
}
BG = {"b": "\033[44m", "k": "\033[40m", "c": "\033[46m", "g": "\033[42m"}
HIDE_CUR = "\033[?25l"
SHOW_CUR = "\033[?25h"
ERASE = "\033[2K"
UP = "\033[A"

def c(fg: str, text: str, bold: bool = False) -> str:
    return f"{BOLD if bold else ''}{FG.get(fg,'')}{text}{RST}"

def _clear() -> None:
    os.system("cls" if platform.system() == "Windows" else "clear")

def _cols() -> int:
    return shutil.get_terminal_size((80, 24)).columns


# ── raw key input ─────────────────────────────────────────────────────────────
def _getch() -> str:
    try:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                return {"[A": "UP", "[B": "DOWN"}.get(seq, "ESC")
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
        raw = input()
        return raw.strip() or "ENTER"


# ── spinner ───────────────────────────────────────────────────────────────────
class Spinner:
    DOTS = ["   ", ".  ", ".. ", "...", " ..", "  .", "   "]
    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
    def __enter__(self) -> "Spinner":
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        sys.stdout.write(HIDE_CUR); sys.stdout.flush()
        self._t.start(); return self
    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._t: self._t.join()
        sys.stdout.write(f"\r{ERASE}{SHOW_CUR}"); sys.stdout.flush()
    def _run(self) -> None:
        for f in itertools.cycle(self.DOTS):
            if self._stop.is_set(): break
            sys.stdout.write(f"\r  {c('C',f)} {c('W',self.label)}"); sys.stdout.flush()
            self._stop.wait(0.12)

def run_with_spinner(cmd: List[str], label: str, cwd: str = "") -> Tuple[int, str, str]:
    with Spinner(label):
        p = subprocess.run(cmd, cwd=cwd or str(REPO_ROOT), capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


# ── box drawing ───────────────────────────────────────────────────────────────
def box(title: str, lines: List[str], color: str = "C", width: int = 0) -> None:
    w = width or max(max((len(l) for l in lines), default=0), len(title)) + 4
    print(f"  {c(color, '┌' + '─' * w + '┐')}")
    if title:
        pad = w - len(title) - 2
        print(f"  {c(color, '│')} {c('W', title, bold=True)}{' ' * pad}{c(color, '│')}")
        print(f"  {c(color, '├' + '─' * w + '┤')}")
    for line in lines:
        pad = w - len(line) - 2
        print(f"  {c(color, '│')} {line}{' ' * max(pad, 0)}{c(color, '│')}")
    print(f"  {c(color, '└' + '─' * w + '┘')}")


def hline(color: str = "k") -> None:
    print(f"  {DIM}{'─' * min(_cols() - 4, 60)}{RST}")


# ── select menu (arrow keys) ─────────────────────────────────────────────────
def select(title: str, options: List[str], descs: Optional[List[str]] = None,
           subtitle: str = "") -> int:
    """Arrow-key menu. Returns 0-based index, -1 on quit."""
    sel = 0
    n = len(options)
    has_desc = descs and len(descs) == n

    def _draw() -> None:
        sys.stdout.write(HIDE_CUR)
        print()
        if title:
            print(f"  {c('C', title, bold=True)}")
        if subtitle:
            print(f"  {DIM}{subtitle}{RST}")
        print()
        for i in range(n):
            if i == sel:
                label = f" {options[i]} "
                desc_str = f"  {descs[i]}" if has_desc else ""
                print(f"  {c('C','>', bold=True)} {BG['b']}{FG['W']}{BOLD}{label}{RST}{c('C', desc_str)}")
            else:
                desc_str = f"  {DIM}{descs[i]}{RST}" if has_desc else ""
                print(f"    {DIM}{options[i]}{RST}{desc_str}")
        print()
        print(f"  {DIM}[Up/Down] move  [Enter] select  [q] back{RST}")

    # count lines drawn
    extra = (1 if title else 0) + (1 if subtitle else 0)
    line_count = n + extra + 3  # blanks + hint

    _draw()
    while True:
        key = _getch()
        if key == "UP" and sel > 0:
            sel -= 1
        elif key == "DOWN" and sel < n - 1:
            sel += 1
        elif key == "ENTER":
            sys.stdout.write(SHOW_CUR); sys.stdout.flush()
            return sel
        elif key in ("Q", "CTRL_C", "CTRL_D", "ESC"):
            sys.stdout.write(SHOW_CUR); sys.stdout.flush()
            return -1
        elif key.isdigit() and 1 <= int(key) <= n:
            sel = int(key) - 1
        # redraw
        for _ in range(line_count):
            sys.stdout.write(f"{UP}{ERASE}")
        sys.stdout.write("\r"); sys.stdout.flush()
        _draw()


# ── step tracker ──────────────────────────────────────────────────────────────
def render_steps(steps: List[Tuple[str, str]]) -> None:
    for i, (label, st) in enumerate(steps):
        n = f"{i+1}/{len(steps)}"
        if st == "running":
            print(f"  {c('C', n, True)}  {c('C','>>>')}  {c('W', label, True)}")
        elif st == "done":
            print(f"  {c('G', n)}  {c('G','[ok]')}  {c('W', label)}")
        elif st == "fail":
            print(f"  {c('R', n)}  {c('R','[!!]')}  {c('R', label)}")
        else:
            print(f"  {DIM}{n}  [ ]  {label}{RST}")

def erase_lines(n: int) -> None:
    for _ in range(n):
        sys.stdout.write(f"{UP}{ERASE}")
    sys.stdout.write("\r"); sys.stdout.flush()


# ── file scanner ──────────────────────────────────────────────────────────────
def scan_csv_files() -> List[Path]:
    """Find CSV files in common locations."""
    found: List[Path] = []
    search_dirs = [
        EXAMPLES_DIR,
        DESKTOP,
        Path.home() / "Downloads",
        Path.home() / "Documents",
        REPO_ROOT,
        DEFAULT_OUT,
    ]
    for d in search_dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.csv"))[:10]:
                if f not in found and f.stat().st_size > 100:
                    found.append(f)
    return found[:15]


def read_csv_columns(path: Path) -> List[str]:
    """Read header row of a CSV file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            return [col.strip() for col in header if col.strip()]
    except Exception:
        return []


def csv_row_count(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh) - 1
    except Exception:
        return 0


# ── pixel logo ────────────────────────────────────────────────────────────────
LOGO = f"""
{c('C','',True)}
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


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENS
# ══════════════════════════════════════════════════════════════════════════════

def screen_home() -> int:
    _clear()
    print(LOGO)
    print(f"    {DIM}Medical ML Data Leakage Prevention Pipeline{RST}")
    print(f"    {DIM}28 Fail-Closed Gates | Publication-Grade Evidence{RST}")
    return select(
        "", 
        ["Quick Start", "Download", "Split CSV", "Full Pipeline", "Health Check", "Guide", "Quit"],
        ["One-click: download + split a real dataset",
         "Get UCI medical datasets",
         "Split your own CSV with safety guarantees",
         "End-to-end training with 28 gates",
         "Verify Python, packages, CLI",
         "Learn about data leakage prevention",
         ""],
    )


# ── Quick Start ───────────────────────────────────────────────────────────────

def action_quick_start() -> None:
    _clear()
    box("QUICK START", [
        "Download a real UCI medical dataset, split it with",
        "patient-disjoint safety, and get ready to train.",
        f"Output: {DEFAULT_OUT}/",
    ], color="G")

    ds = select(
        "Pick a dataset",
        ["Heart Disease", "Breast Cancer", "Kidney Disease"],
        ["UCI Cleveland -- 297 patients, 13 features, predict heart disease",
         "UCI Wisconsin -- 569 patients, 30 features, malignant vs benign",
         "UCI CKD -- 399 patients, 24 features, predict chronic kidney disease"],
    )
    if ds < 0: return
    ds_keys = ["heart", "breast", "ckd"]
    ds_files = ["heart_disease", "breast_cancer", "chronic_kidney_disease"]
    ds_labels = ["Heart Disease", "Breast Cancer", "Kidney Disease"]
    key, fname, label = ds_keys[ds], ds_files[ds], ds_labels[ds]

    steps: List[Tuple[str, str]] = [
        (f"Download {label} from UCI", "pending"),
        ("Split into train / valid / test", "pending"),
        ("Verify patient-disjoint + temporal order", "pending"),
    ]
    print()
    steps[0] = (steps[0][0], "running")
    render_steps(steps)

    rc, _, err = run_with_spinner(
        [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"), key],
        f"Downloading {label}...",
    )
    erase_lines(len(steps))
    if rc != 0:
        steps[0] = (steps[0][0], "fail")
        render_steps(steps)
        print(f"\n  {c('R','Download failed.')}")
        if err:
            for l in err.strip().split("\n")[-3:]: print(f"  {DIM}{l}{RST}")
        return
    steps[0] = (steps[0][0], "done")

    csv_path = EXAMPLES_DIR / f"{fname}.csv"
    out_dir = DEFAULT_OUT / fname
    steps[1] = (steps[1][0], "running")
    render_steps(steps)

    rc, _, err = run_with_spinner(
        [sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
         "--input", str(csv_path), "--output-dir", str(out_dir / "data"),
         "--patient-id-col", "patient_id", "--target-col", "y",
         "--time-col", "event_time", "--strategy", "grouped_temporal"],
        "Splitting with safety checks...",
    )
    erase_lines(len(steps))
    if rc != 0:
        steps[1] = (steps[1][0], "fail")
        render_steps(steps)
        print(f"\n  {c('R','Split failed.')}")
        if err:
            for l in err.strip().split("\n")[-3:]: print(f"  {DIM}{l}{RST}")
        return
    steps[1] = (steps[1][0], "done")
    steps[2] = (steps[2][0], "done")
    render_steps(steps)

    # Show results
    try:
        import pandas as pd
        train = pd.read_csv(out_dir / "data" / "train.csv")
        valid = pd.read_csv(out_dir / "data" / "valid.csv")
        test = pd.read_csv(out_dir / "data" / "test.csv")
        print()
        box("Results", [
            f"train.csv   {len(train):>4} rows   {train['patient_id'].nunique():>4} patients",
            f"valid.csv   {len(valid):>4} rows   {valid['patient_id'].nunique():>4} patients",
            f"test.csv    {len(test):>4} rows    {test['patient_id'].nunique():>3} patients",
            "",
            f"Saved to: {out_dir}/data/",
        ], color="G")
    except Exception:
        print(f"\n  {c('G','Done!',True)} Output: {out_dir}/data/")

    print()
    print(f"  {DIM}Next: select 'Full Pipeline' to train a model.{RST}")


# ── Download ──────────────────────────────────────────────────────────────────

def action_download() -> None:
    _clear()
    box("DOWNLOAD DATASET", [
        "Download real UCI medical datasets.",
        f"Files saved to: {EXAMPLES_DIR}/",
    ], color="Y")

    ch = select(
        "Which dataset?",
        ["Heart Disease", "Breast Cancer", "Kidney Disease", "All three"],
        ["UCI Cleveland -- 297 rows, 13 features",
         "UCI Wisconsin -- 569 rows, 30 features",
         "UCI CKD -- 399 rows, 24 features",
         "Download all datasets at once"],
    )
    if ch < 0: return
    key = ["heart", "breast", "ckd", "all"][ch]

    rc, out, err = run_with_spinner(
        [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"), key],
        f"Downloading {key}...",
    )
    if rc == 0:
        info = [l.strip() for l in (out or "").split("\n") if "Output:" in l or "Rows:" in l]
        box("Download Complete", info or ["Done!"], color="G")
    else:
        print(f"  {c('R','[!!]',True)} Download failed.")
        if err:
            for l in err.strip().split("\n")[-3:]: print(f"  {DIM}{l}{RST}")


# ── Split CSV ─────────────────────────────────────────────────────────────────

def action_split() -> None:
    _clear()
    box("SPLIT YOUR CSV", [
        "Split a CSV file into train/valid/test sets.",
        "Patient-disjoint, with medical safety guarantees.",
    ], color="M")

    # Step 1: Find CSV files automatically
    csv_files = scan_csv_files()
    if csv_files:
        names = [f"{f.name}" for f in csv_files]
        descs = [f"{csv_row_count(f)} rows  --  {f.parent}" for f in csv_files]
        names.append("Enter path manually...")
        descs.append("")

        fi = select("Select your CSV file", names, descs)
        if fi < 0: return
        if fi == len(csv_files):
            sys.stdout.write(SHOW_CUR)
            csv_path = input(f"  {c('C','>')} {c('W','CSV path')}: ").strip()
        else:
            csv_path = str(csv_files[fi])
    else:
        sys.stdout.write(SHOW_CUR)
        csv_path = input(f"  {c('C','>')} {c('W','CSV path')}: ").strip()

    if not csv_path or not Path(csv_path).exists():
        print(f"  {c('R','File not found.')}")
        return

    # Step 2: Read columns and let user pick
    columns = read_csv_columns(Path(csv_path))
    if not columns:
        print(f"  {c('R','Cannot read CSV header.')}")
        return

    row_count = csv_row_count(Path(csv_path))
    print(f"\n  {c('W', Path(csv_path).name, bold=True)}  {DIM}{row_count} rows, {len(columns)} columns{RST}")
    hline()

    # Patient ID
    pi = select("Which column is the Patient ID?", columns)
    if pi < 0: return
    pid_col = columns[pi]

    # Target
    remaining = [col for col in columns if col != pid_col]
    ti = select("Which column is the Target (0/1)?", remaining)
    if ti < 0: return
    target_col = remaining[ti]

    # Strategy
    si = select(
        "Split strategy",
        ["Grouped Temporal", "Grouped Random", "Stratified Grouped"],
        ["Sort by time, patient-disjoint (recommended)",
         "Random patient-disjoint split",
         "Preserve positive rate across splits"],
    )
    if si < 0: return
    strategy = ["grouped_temporal", "grouped_random", "stratified_grouped"][si]

    # Time column (if temporal)
    time_col = ""
    if strategy == "grouped_temporal":
        time_remaining = [col for col in columns if col not in (pid_col, target_col)]
        tci = select("Which column is the Time/Date?", time_remaining)
        if tci < 0: return
        time_col = time_remaining[tci]

    out_dir = DEFAULT_OUT / Path(csv_path).stem

    # Confirm
    print()
    box("Configuration", [
        f"File:     {Path(csv_path).name}  ({row_count} rows)",
        f"Patient:  {pid_col}",
        f"Target:   {target_col}",
        f"Time:     {time_col or '(none)'}",
        f"Strategy: {strategy}",
        f"Output:   {out_dir}/data/",
    ], color="B")

    ci = select("", ["Run split", "Cancel"])
    if ci != 0: return

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
        "--input", csv_path, "--output-dir", str(out_dir / "data"),
        "--patient-id-col", pid_col, "--target-col", target_col,
        "--strategy", strategy,
    ]
    if time_col:
        cmd.extend(["--time-col", time_col])

    rc, out, err = run_with_spinner(cmd, "Splitting...")
    if rc == 0:
        try:
            import pandas as pd
            tr = pd.read_csv(out_dir / "data" / "train.csv")
            va = pd.read_csv(out_dir / "data" / "valid.csv")
            te = pd.read_csv(out_dir / "data" / "test.csv")
            box("Split Complete", [
                f"train.csv   {len(tr):>4} rows",
                f"valid.csv   {len(va):>4} rows",
                f"test.csv    {len(te):>4} rows",
                "", f"Saved to: {out_dir}/data/",
            ], color="G")
        except Exception:
            print(f"  {c('G','[ok]',True)} Split complete! Output: {out_dir}/data/")
    else:
        print(f"  {c('R','[!!]',True)} Split failed.")
        if err:
            for l in err.strip().split("\n")[-5:]: print(f"  {DIM}{l}{RST}")


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def action_full_pipeline() -> None:
    _clear()
    box("FULL PIPELINE", [
        "End-to-end training with 28 fail-closed safety gates.",
        "Generates publication-grade evidence artifacts.",
    ], color="Y")

    mi = select(
        "Mode",
        ["Demo Mode", "Your CSV"],
        ["Use synthetic data -- great for first run",
         "Bring your own dataset"],
    )
    if mi < 0: return

    project_root = str(DEFAULT_OUT / "pipeline")

    if mi == 0:
        # Demo
        print()
        box("Demo Pipeline", [
            f"Output: {project_root}/",
            "This will take 3-8 minutes.",
        ], color="C")
        ci = select("", ["Start demo pipeline", "Cancel"])
        if ci != 0: return

        rc, _, err = run_with_spinner(
            [sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
             "--project-root", project_root, "--mode", "guided", "--yes"],
            "Running full pipeline...",
        )
    else:
        # User CSV -- same selection flow as Split
        csv_files = scan_csv_files()
        if csv_files:
            names = [f.name for f in csv_files]
            descs = [f"{csv_row_count(f)} rows -- {f.parent}" for f in csv_files]
            names.append("Enter path manually...")
            descs.append("")
            fi = select("Select your CSV", names, descs)
            if fi < 0: return
            if fi == len(csv_files):
                sys.stdout.write(SHOW_CUR)
                csv_path = input(f"  {c('C','>')} {c('W','CSV path')}: ").strip()
            else:
                csv_path = str(csv_files[fi])
        else:
            sys.stdout.write(SHOW_CUR)
            csv_path = input(f"  {c('C','>')} {c('W','CSV path')}: ").strip()

        if not csv_path or not Path(csv_path).exists():
            print(f"  {c('R','File not found.')}"); return

        columns = read_csv_columns(Path(csv_path))
        if not columns:
            print(f"  {c('R','Cannot read CSV.')}"); return

        pi = select("Patient ID column?", columns)
        if pi < 0: return
        pid_col = columns[pi]

        remaining = [col for col in columns if col != pid_col]
        ti = select("Target column (0/1)?", remaining)
        if ti < 0: return
        target_col = remaining[ti]

        si = select("Split strategy",
                     ["Grouped Temporal", "Grouped Random", "Stratified Grouped"],
                     ["Sort by time (recommended)", "Random split", "Preserve positive rate"])
        if si < 0: return
        strategy = ["grouped_temporal", "grouped_random", "stratified_grouped"][si]

        time_col = ""
        if strategy == "grouped_temporal":
            time_remaining = [col for col in columns if col not in (pid_col, target_col)]
            tci = select("Time column?", time_remaining)
            if tci < 0: return
            time_col = time_remaining[tci]

        project_root = str(DEFAULT_OUT / Path(csv_path).stem / "pipeline")
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
            "--project-root", project_root, "--mode", "guided", "--yes",
            "--input-csv", csv_path,
            "--patient-id-col", pid_col, "--target-col", target_col,
            "--split-strategy", strategy,
        ]
        if time_col:
            cmd.extend(["--time-col", time_col])

        print()
        box("Pipeline Configuration", [
            f"File:     {Path(csv_path).name}",
            f"Patient:  {pid_col}  |  Target: {target_col}  |  Time: {time_col or '(none)'}",
            f"Strategy: {strategy}",
            f"Output:   {project_root}/",
        ], color="C")
        ci = select("", ["Start pipeline", "Cancel"])
        if ci != 0: return

        rc, _, err = run_with_spinner(cmd, "Running full pipeline...")

    if rc == 0:
        box("Pipeline Complete", [
            f"Results: {project_root}/",
            "  evidence/  -- audit artifacts",
            "  models/    -- trained model",
            "  data/      -- split datasets",
        ], color="G")
    else:
        print(f"  {c('R','[!!]',True)} Pipeline had failures.")
        if err:
            for l in (err or "").strip().split("\n")[-5:]: print(f"  {DIM}{l}{RST}")


# ── Health Check ──────────────────────────────────────────────────────────────

def action_health_check() -> None:
    _clear()
    box("HEALTH CHECK", ["Verifying your environment..."], color="G")
    print()

    checks: List[Tuple[str, bool, str]] = []
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append((f"Python {py}", sys.version_info >= (3, 9), ""))

    for pkg in ["numpy", "pandas", "sklearn", "joblib"]:
        try:
            mod = __import__(pkg)
            checks.append((f"{pkg} {getattr(mod,'__version__','?')}", True, ""))
        except ImportError:
            checks.append((pkg, False, "required"))

    for pkg, label in [("xgboost","XGBoost"),("catboost","CatBoost"),("lightgbm","LightGBM")]:
        try:
            mod = __import__(pkg)
            checks.append((f"{label} {getattr(mod,'__version__','?')}", True, "optional"))
        except ImportError:
            checks.append((label, False, "optional"))

    rc, _, _ = run_with_spinner(
        [sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "--help"], "Checking CLI...")
    checks.append(("mlgg.py CLI", rc == 0, ""))

    for name, ok, note in checks:
        icon = c('G','[ok]') if ok else c('R','[--]')
        extra = f"  {DIM}{note}{RST}" if note else ""
        print(f"  {icon}  {name}{extra}")

    passed = sum(ok for _, ok, _ in checks)
    total = len(checks)
    w = 25
    filled = int(w * passed / total)
    print(f"\n  {c('G','#' * filled)}{DIM}{'.' * (w - filled)}{RST}  {passed}/{total}\n")

    ci = select("", ["Run full doctor", "Back"])
    if ci == 0:
        print(f"  {DIM}$ python3 scripts/mlgg.py doctor{RST}\n")
        subprocess.run([sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "doctor"],
                       cwd=str(REPO_ROOT), text=True)


# ── Guide ─────────────────────────────────────────────────────────────────────

def action_guide() -> None:
    _clear()
    pages = [
        ("What is Data Leakage?", [
            "Data leakage in medical ML means information from",
            "outside the intended training scope accidentally",
            "influences model training. This inflates performance",
            "and can lead to unsafe clinical decisions.",
        ]),
        ("What This Pipeline Does", [
            "- 28 sequential fail-closed safety gates",
            "- Patient-disjoint temporal splitting",
            "- Feature leakage detection",
            "- Tuning and calibration leakage guards",
            "- Publication-grade evidence artifacts",
        ]),
        ("Available Datasets", [
            "Heart Disease   -- 297 rows, 13 features",
            "Breast Cancer   -- 569 rows, 30 features",
            "Kidney Disease  -- 399 rows, 24 features",
            "",
            "Download: python3 examples/download_real_data.py heart",
        ]),
        ("Getting Started", [
            "git clone https://github.com/Furinaaa-Cancan/",
            "    medical-ml-leakage-guard.git",
            "cd medical-ml-leakage-guard",
            "pip install -r requirements.txt",
            "python3 scripts/mlgg.py play",
        ]),
    ]

    page = 0
    while 0 <= page < len(pages):
        _clear()
        title, lines = pages[page]
        box(f"GUIDE ({page+1}/{len(pages)})", [], color="B")
        print()
        print(f"  {c('Y', title, bold=True)}")
        print()
        for l in lines:
            print(f"    {l}")
        print()
        nav = ["Next page"] if page < len(pages) - 1 else []
        nav += (["Previous page"] if page > 0 else []) + ["Back to menu"]
        ni = select("", nav)
        if ni < 0 or nav[ni] == "Back to menu":
            break
        elif nav[ni] == "Next page":
            page += 1
        elif nav[ni] == "Previous page":
            page -= 1


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__); return 0

    while True:
        try:
            ch = screen_home()
        except KeyboardInterrupt:
            print(f"\n\n  {c('C','Bye!')}"); return 0

        if ch < 0 or ch == 6:
            print(f"\n  {c('C','Bye!')}\n"); return 0

        try:
            [action_quick_start, action_download, action_split,
             action_full_pipeline, action_health_check, action_guide][ch]()
        except KeyboardInterrupt:
            print(f"\n  {DIM}Interrupted.{RST}"); continue

        print()
        sys.stdout.write(SHOW_CUR)
        try:
            input(f"  {DIM}Press Enter to return to menu...{RST}")
        except (EOFError, KeyboardInterrupt):
            return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        sys.stdout.write(SHOW_CUR); sys.stdout.flush()
