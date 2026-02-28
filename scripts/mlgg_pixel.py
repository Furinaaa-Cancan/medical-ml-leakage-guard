#!/usr/bin/env python3
"""
ML Leakage Guard — Pixel-Art Interactive CLI Launcher.

A retro pixel-art themed interactive terminal experience for the
ml-leakage-guard medical ML pipeline. Guides users through dataset
download, splitting, training, and full pipeline execution with
colorful visuals and step-by-step interaction.

Usage:
    python3 scripts/mlgg_pixel.py
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Terminal Colors (ANSI) ───────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
BLINK = "\033[5m"

# Foreground
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Bright foreground
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

# Background
BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# ─── Pixel Art Assets ─────────────────────────────────────────────────────────

LOGO_ART = rf"""
{BRIGHT_CYAN}    ██╗     ███████╗ █████╗ ██╗  ██╗ █████╗  ██████╗ ███████╗
    ██║     ██╔════╝██╔══██╗██║ ██╔╝██╔══██╗██╔════╝ ██╔════╝
    ██║     █████╗  ███████║█████╔╝ ███████║██║  ███╗█████╗
    ██║     ██╔══╝  ██╔══██║██╔═██╗ ██╔══██║██║   ██║██╔══╝
    ███████╗███████╗██║  ██║██║  ██╗██║  ██║╚██████╔╝███████╗
    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝{RESET}

{BRIGHT_YELLOW}     ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
    ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
    ██║  ███╗██║   ██║███████║██████╔╝██║  ██║
    ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
    ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
     ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ {RESET}
"""

SHIELD_ART = rf"""
{BRIGHT_GREEN}         ▄▄████████▄▄
       ▄██{BRIGHT_WHITE}████████████{BRIGHT_GREEN}██▄
      ███{BRIGHT_WHITE}██{BRIGHT_RED}╔══════╗{BRIGHT_WHITE}██{BRIGHT_GREEN}███
      ███{BRIGHT_WHITE}██{BRIGHT_RED}║ MLGG ║{BRIGHT_WHITE}██{BRIGHT_GREEN}███
      ███{BRIGHT_WHITE}██{BRIGHT_RED}║ v1.0 ║{BRIGHT_WHITE}██{BRIGHT_GREEN}███
      ███{BRIGHT_WHITE}██{BRIGHT_RED}╚══════╝{BRIGHT_WHITE}██{BRIGHT_GREEN}███
       ███{BRIGHT_WHITE}████████████{BRIGHT_GREEN}███
        ████{BRIGHT_WHITE}████████{BRIGHT_GREEN}████
          ████{BRIGHT_WHITE}████{BRIGHT_GREEN}████
            ████████
              ████
               ██{RESET}
"""

HEART_ART = rf"""
{BRIGHT_RED}      ████   ████
    ██████ ██████
   ████████████████
   ████████████████
    ██████████████
      ██████████
        ██████
          ██{RESET}
"""

TROPHY_ART = rf"""
{BRIGHT_YELLOW}       ▄████████▄
      █{BRIGHT_WHITE}██████████{BRIGHT_YELLOW}█
      █{BRIGHT_WHITE}██████████{BRIGHT_YELLOW}█
       ██████████
        ████████
         ██████
          ████
        ████████
       ██████████{RESET}
"""


# ─── Utility Functions ────────────────────────────────────────────────────────


def clear_screen() -> None:
    os.system("cls" if platform.system() == "Windows" else "clear")


def term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def center(text: str, width: int = 0) -> str:
    w = width or term_width()
    lines = text.split("\n")
    return "\n".join(line.center(w) if line.strip() else line for line in lines)


def print_centered(text: str) -> None:
    print(center(text))


def print_box(text: str, color: str = BRIGHT_CYAN, padding: int = 2) -> None:
    lines = text.split("\n")
    max_len = max(len(line) for line in lines)
    w = max_len + padding * 2

    print(f"{color}  ╔{'═' * w}╗{RESET}")
    for line in lines:
        padded = line.ljust(max_len)
        print(f"{color}  ║{' ' * padding}{BRIGHT_WHITE}{padded}{color}{' ' * padding}║{RESET}")
    print(f"{color}  ╚{'═' * w}╝{RESET}")


def print_divider(char: str = "─", color: str = DIM) -> None:
    w = min(term_width(), 72)
    print(f"{color}  {'  ' + char * (w - 4)}{RESET}")


def type_text(text: str, delay: float = 0.02, color: str = "") -> None:
    prefix = color if color else ""
    suffix = RESET if color else ""
    for ch in text:
        sys.stdout.write(f"{prefix}{ch}{suffix}")
        sys.stdout.flush()
        time.sleep(delay)
    print()


def pixel_progress(current: int, total: int, width: int = 30, label: str = "") -> str:
    filled = int(width * current / total) if total > 0 else 0
    empty = width - filled
    bar = f"{BRIGHT_GREEN}{'█' * filled}{DIM}{'░' * empty}{RESET}"
    pct = int(100 * current / total) if total > 0 else 0
    lbl = f" {label}" if label else ""
    return f"  {bar} {BRIGHT_WHITE}{pct:3d}%{RESET}{lbl}"


def show_progress_animation(steps: List[str], delay: float = 0.4) -> None:
    for i, step in enumerate(steps):
        bar = pixel_progress(i + 1, len(steps), label=step)
        print(f"\r{bar}", end="", flush=True)
        time.sleep(delay)
    print()


def prompt_choice(options: List[Tuple[str, str]], prompt_text: str = "Choose") -> int:
    print()
    for i, (icon, label) in enumerate(options):
        num_color = BRIGHT_YELLOW if i < len(options) - 1 else DIM
        print(f"  {num_color}[{i + 1}]{RESET}  {icon}  {BRIGHT_WHITE}{label}{RESET}")
    print()
    while True:
        try:
            raw = input(f"  {BRIGHT_CYAN}▶ {prompt_text} [1-{len(options)}]: {RESET}").strip()
            if not raw:
                continue
            n = int(raw)
            if 1 <= n <= len(options):
                return n
        except (ValueError, EOFError):
            pass
        print(f"  {RED}  Invalid choice. Try again.{RESET}")


def prompt_input(label: str, default: str = "", required: bool = True) -> str:
    suffix = f" {DIM}[{default}]{RESET}" if default else ""
    while True:
        try:
            raw = input(f"  {BRIGHT_CYAN}▶ {label}{suffix}: {RESET}").strip()
        except EOFError:
            raw = ""
        value = raw if raw else default
        if value or not required:
            return value
        print(f"  {RED}  This field is required.{RESET}")


def prompt_confirm(label: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        raw = input(f"  {BRIGHT_CYAN}▶ {label} [{hint}]: {RESET}").strip().lower()
    except EOFError:
        raw = ""
    if not raw:
        return default
    return raw in ("y", "yes")


def run_live(cmd: List[str], cwd: str = "") -> int:
    print(f"\n  {DIM}$ {shlex.join(cmd)}{RESET}\n")
    proc = subprocess.run(
        cmd,
        cwd=cwd or str(REPO_ROOT),
        text=True,
    )
    return proc.returncode


def run_capture(cmd: List[str], cwd: str = "") -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd or str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def print_status(ok: bool, message: str) -> None:
    if ok:
        print(f"  {BRIGHT_GREEN}██ PASS{RESET}  {message}")
    else:
        print(f"  {BRIGHT_RED}██ FAIL{RESET}  {message}")


def show_sparkle(text: str) -> None:
    frames = ["✦", "✧", "✦", "✧", "★"]
    for frame in frames:
        sys.stdout.write(f"\r  {BRIGHT_YELLOW}{frame} {text} {frame}{RESET}  ")
        sys.stdout.flush()
        time.sleep(0.15)
    print()


# ─── Screens ──────────────────────────────────────────────────────────────────


def show_splash() -> None:
    clear_screen()
    print(LOGO_ART)
    print_centered(f"{DIM}Medical ML Data Leakage Prevention Pipeline{RESET}")
    print_centered(f"{DIM}28 Fail-Closed Safety Gates • Publication-Grade{RESET}")
    print()
    print_divider("═", BRIGHT_CYAN)
    print()


def show_main_menu() -> int:
    show_splash()
    print_box(
        "Welcome, Researcher!\n"
        "Choose an action to get started.\n"
        "All operations enforce strict leakage controls.",
        color=BRIGHT_BLUE,
    )
    return prompt_choice(
        [
            ("🚀", "Quick Start  — Download dataset + split + ready to train"),
            ("📊", "Download Dataset  — Get a real UCI medical dataset"),
            ("✂️ ", "Split Data  — Split your CSV with safety guarantees"),
            ("🏥", "Full Pipeline  — Run complete onboarding (demo or your data)"),
            ("🔍", "Health Check  — Verify installation & dependencies"),
            ("📖", "Guide & Info  — Learn about this pipeline"),
            ("🚪", "Exit"),
        ],
        prompt_text="Select",
    )


# ─── Action: Quick Start ─────────────────────────────────────────────────────


def action_quick_start() -> None:
    clear_screen()
    print(SHIELD_ART)
    print_box("QUICK START", color=BRIGHT_GREEN)
    type_text("  Let's get you up and running with a real medical dataset!", color=BRIGHT_WHITE)
    print()

    # Step 1: Choose dataset
    print(f"  {BRIGHT_YELLOW}STEP 1{RESET}  {BRIGHT_WHITE}Choose a dataset{RESET}")
    print_divider()
    ds_choice = prompt_choice(
        [
            ("❤️ ", "Heart Disease (Cleveland) — 297 rows, 13 features"),
            ("🎀", "Breast Cancer (Wisconsin) — 569 rows, 30 features"),
            ("🫘", "Chronic Kidney Disease — 399 rows, 24 features"),
        ],
        prompt_text="Dataset",
    )
    ds_map = {1: "heart", 2: "breast", 3: "ckd"}
    ds_name_map = {1: "heart_disease", 2: "breast_cancer", 3: "chronic_kidney_disease"}
    ds_key = ds_map[ds_choice]
    ds_file = ds_name_map[ds_choice]

    # Step 2: Download
    print()
    print(f"  {BRIGHT_YELLOW}STEP 2{RESET}  {BRIGHT_WHITE}Downloading dataset...{RESET}")
    print_divider()
    show_progress_animation(
        ["Checking local files...", "Parsing raw data...", "Converting format...", "Adding metadata..."],
        delay=0.3,
    )

    rc = run_live(
        [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"), ds_key],
    )
    if rc != 0:
        print_status(False, "Download failed. Check error messages above.")
        return
    print_status(True, "Dataset downloaded successfully!")

    # Step 3: Split
    csv_path = EXAMPLES_DIR / f"{ds_file}.csv"
    output_dir = Path(f"/tmp/mlgg_pixel_{ds_file}")
    print()
    print(f"  {BRIGHT_YELLOW}STEP 3{RESET}  {BRIGHT_WHITE}Splitting with safety guarantees...{RESET}")
    print_divider()
    show_progress_animation(
        [
            "Loading CSV...",
            "Validating columns...",
            "Checking binary target...",
            "Temporal grouping...",
            "Patient-disjoint split...",
            "Post-split validation...",
        ],
        delay=0.25,
    )

    rc = run_live(
        [
            sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
            "--input", str(csv_path),
            "--output-dir", str(output_dir / "data"),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--time-col", "event_time",
            "--strategy", "grouped_temporal",
        ],
    )
    if rc != 0:
        print_status(False, "Split failed. Check error messages above.")
        return

    print()
    show_sparkle("ALL CHECKS PASSED")
    print()
    print_box(
        f"Dataset ready at: {output_dir}\n"
        f"\n"
        f"  train.csv  — Training set\n"
        f"  valid.csv  — Validation set\n"
        f"  test.csv   — Test set\n"
        f"\n"
        f"Next: run the full pipeline with option [4]",
        color=BRIGHT_GREEN,
    )


# ─── Action: Download Dataset ────────────────────────────────────────────────


def action_download() -> None:
    clear_screen()
    print(HEART_ART)
    print_box("DOWNLOAD REAL DATASET", color=BRIGHT_RED)
    print()
    print(f"  {BRIGHT_WHITE}Available UCI Medical Datasets:{RESET}")
    print()

    ds_choice = prompt_choice(
        [
            ("❤️ ", "Heart Disease (Cleveland) — 297 rows, 13 features, predict heart disease"),
            ("🎀", "Breast Cancer (Wisconsin) — 569 rows, 30 features, malignant vs benign"),
            ("🫘", "Chronic Kidney Disease — 399 rows, 24 features, predict CKD"),
            ("📦", "All datasets"),
        ],
        prompt_text="Dataset",
    )
    ds_map = {1: "heart", 2: "breast", 3: "ckd", 4: "all"}
    ds_key = ds_map[ds_choice]

    print()
    type_text(f"  Downloading {ds_key}...", delay=0.03, color=BRIGHT_CYAN)
    rc = run_live([sys.executable, str(EXAMPLES_DIR / "download_real_data.py"), ds_key])

    if rc == 0:
        print()
        show_sparkle("Download Complete!")
    else:
        print_status(False, "Download failed.")


# ─── Action: Split Data ──────────────────────────────────────────────────────


def action_split() -> None:
    clear_screen()
    print_box("SPLIT DATA", color=BRIGHT_MAGENTA)
    print()
    type_text("  Split your CSV into train/valid/test with medical safety guarantees.", color=BRIGHT_WHITE)
    print()

    # Input CSV
    csv_path = prompt_input("Path to your CSV file")
    if not Path(csv_path).exists():
        print_status(False, f"File not found: {csv_path}")
        return

    # Column names
    print()
    print(f"  {BRIGHT_YELLOW}Column Configuration{RESET}")
    print_divider()
    pid_col = prompt_input("Patient ID column", default="patient_id")
    target_col = prompt_input("Target column (binary 0/1)", default="y")

    # Strategy
    print()
    print(f"  {BRIGHT_YELLOW}Split Strategy{RESET}")
    print_divider()
    strat_choice = prompt_choice(
        [
            ("⏰", "Grouped Temporal  — Sort by time, recommended for longitudinal data"),
            ("🎲", "Grouped Random  — Random split, for cross-sectional data"),
            ("📊", "Stratified Grouped  — Preserve positive rate across splits"),
        ],
        prompt_text="Strategy",
    )
    strat_map = {1: "grouped_temporal", 2: "grouped_random", 3: "stratified_grouped"}
    strategy = strat_map[strat_choice]

    time_col = ""
    if strategy == "grouped_temporal":
        time_col = prompt_input("Time column", default="event_time")

    # Output
    output_dir = prompt_input("Output directory", default="/tmp/mlgg_split/data")

    # Confirm
    print()
    print_box(
        f"Input:    {csv_path}\n"
        f"Strategy: {strategy}\n"
        f"Patient:  {pid_col}\n"
        f"Target:   {target_col}\n"
        f"Time:     {time_col or '(none)'}\n"
        f"Output:   {output_dir}",
        color=BRIGHT_BLUE,
    )
    if not prompt_confirm("Proceed with split?"):
        print(f"  {DIM}Cancelled.{RESET}")
        return

    # Run
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

    print()
    show_progress_animation(
        ["Validating input...", "Checking columns...", "Splitting...", "Post-split checks..."],
        delay=0.3,
    )
    rc = run_live(cmd)

    if rc == 0:
        show_sparkle("Split Complete!")
    else:
        print_status(False, "Split failed. Check the errors above.")


# ─── Action: Full Pipeline ───────────────────────────────────────────────────


def action_full_pipeline() -> None:
    clear_screen()
    print(TROPHY_ART)
    print_box("FULL PIPELINE", color=BRIGHT_YELLOW)
    print()
    type_text("  Run the complete publication-grade ML pipeline.", color=BRIGHT_WHITE)
    print()

    mode_choice = prompt_choice(
        [
            ("🎮", "Demo Mode  — Use synthetic data, great for first-time users"),
            ("📂", "Your Data  — Use your own CSV file"),
        ],
        prompt_text="Mode",
    )

    project_root = prompt_input("Project output directory", default="/tmp/mlgg_pipeline")

    if mode_choice == 1:
        # Demo mode
        print()
        print(f"  {BRIGHT_YELLOW}Running full demo pipeline...{RESET}")
        print(f"  {DIM}This may take 3-8 minutes. Grab a coffee! ☕{RESET}")
        print()

        show_progress_animation(
            [
                "Step 1: Environment check...",
                "Step 2: Initialize project...",
                "Step 3: Generate demo data...",
                "Step 4: Align configurations...",
                "Step 5: Train & evaluate...",
                "Step 6: Generate attestation...",
                "Step 7: Run safety gates...",
                "Step 8: Final validation...",
            ],
            delay=0.2,
        )

        rc = run_live([
            sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
            "--project-root", project_root,
            "--mode", "guided",
            "--yes",
        ])
    else:
        # User data mode
        csv_path = prompt_input("Path to your CSV file")
        pid_col = prompt_input("Patient ID column", default="patient_id")
        target_col = prompt_input("Target column", default="y")
        time_col = prompt_input("Time column (leave empty if none)", default="", required=False)

        strategy_choice = prompt_choice(
            [
                ("⏰", "Grouped Temporal"),
                ("🎲", "Grouped Random"),
                ("📊", "Stratified Grouped"),
            ],
            prompt_text="Strategy",
        )
        strat_map = {1: "grouped_temporal", 2: "grouped_random", 3: "stratified_grouped"}
        strategy = strat_map[strategy_choice]

        cmd = [
            sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
            "--project-root", project_root,
            "--mode", "guided",
            "--yes",
            "--input-csv", csv_path,
            "--patient-id-col", pid_col,
            "--target-col", target_col,
            "--split-strategy", strategy,
        ]
        if time_col:
            cmd.extend(["--time-col", time_col])

        print()
        print(f"  {BRIGHT_YELLOW}Running full pipeline on your data...{RESET}")
        print(f"  {DIM}This may take several minutes.{RESET}")
        print()
        rc = run_live(cmd)

    if rc == 0:
        print()
        show_sparkle("PIPELINE COMPLETE!")
        print()
        print_box(
            f"Results at: {project_root}\n"
            f"\n"
            f"  evidence/  — All audit artifacts\n"
            f"  configs/   — Pipeline configuration\n"
            f"  models/    — Trained model\n"
            f"  data/      — Split datasets\n"
            f"\n"
            f"  Check: evidence/onboarding_report.json\n"
            f"  Check: evidence/user_summary.md",
            color=BRIGHT_GREEN,
        )
    else:
        print_status(False, "Pipeline had failures. Check the output above.")


# ─── Action: Health Check ────────────────────────────────────────────────────


def action_health_check() -> None:
    clear_screen()
    print_box("HEALTH CHECK", color=BRIGHT_GREEN)
    print()
    type_text("  Verifying your environment...", color=BRIGHT_CYAN)
    print()

    checks: List[Tuple[str, bool, str]] = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 9)
    checks.append((f"Python {py_ver}", ok, "Need Python >= 3.9"))

    # Core packages
    for pkg in ["numpy", "pandas", "sklearn", "joblib"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            checks.append((f"{pkg} {ver}", True, ""))
        except ImportError:
            checks.append((f"{pkg}", False, "Not installed — pip install -r requirements.txt"))

    # Optional packages
    for pkg, label in [("xgboost", "XGBoost"), ("catboost", "CatBoost"), ("lightgbm", "LightGBM")]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            checks.append((f"{label} {ver}", True, "(optional)"))
        except ImportError:
            checks.append((f"{label}", False, "(optional — enhanced model families)"))

    # CLI
    rc, stdout, _ = run_capture([sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "--help"])
    checks.append(("mlgg.py CLI", rc == 0, "" if rc == 0 else "CLI failed"))

    # Print results
    for label, ok, note in checks:
        icon = f"{BRIGHT_GREEN}██{RESET}" if ok else f"{BRIGHT_RED}░░{RESET}"
        note_str = f"  {DIM}{note}{RESET}" if note else ""
        print(f"  {icon}  {BRIGHT_WHITE}{label}{RESET}{note_str}")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print()
    print(pixel_progress(passed, total, label=f"{passed}/{total} checks passed"))
    print()

    # Run doctor
    if prompt_confirm("Run full environment doctor?"):
        run_live([sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "doctor"])


# ─── Action: Guide ───────────────────────────────────────────────────────────


def action_guide() -> None:
    clear_screen()
    print(SHIELD_ART)
    print_box("ML LEAKAGE GUARD — GUIDE", color=BRIGHT_BLUE)
    print()

    sections = [
        (
            "What is Data Leakage?",
            "Data leakage in medical ML means information from outside the\n"
            "intended training scope (test labels, future timestamps, disease-\n"
            "defining variables) accidentally influences model training.\n"
            "This inflates reported performance and can lead to unsafe\n"
            "clinical decisions.",
        ),
        (
            "What This Pipeline Does",
            "• Builds medical binary prediction pipelines under strict controls\n"
            "• Enforces 28 sequential fail-closed safety gates\n"
            "• Covers: split contamination, feature leakage, tuning leakage,\n"
            "  calibration misuse, external cohort robustness, and more\n"
            "• Generates publication-grade evidence artifacts",
        ),
        (
            "Quick Start (Terminal)",
            "  git clone https://github.com/Furinaaa-Cancan/\n"
            "    medical-ml-leakage-guard.git\n"
            "  cd medical-ml-leakage-guard\n"
            "  pip install -r requirements.txt\n"
            "  python3 scripts/mlgg_pixel.py     # This launcher!\n"
            "  # Or: python3 scripts/mlgg.py onboarding --yes",
        ),
        (
            "Available Datasets",
            "  ❤️  Heart Disease (Cleveland)   — 297 rows, 13 features\n"
            "  🎀  Breast Cancer (Wisconsin)   — 569 rows, 30 features\n"
            "  🫘  Chronic Kidney Disease       — 399 rows, 24 features\n"
            "\n"
            "  Download: python3 examples/download_real_data.py heart",
        ),
    ]

    for title, body in sections:
        print(f"  {BRIGHT_YELLOW}■ {title}{RESET}")
        print()
        for line in body.split("\n"):
            print(f"    {BRIGHT_WHITE}{line}{RESET}")
        print()
        print_divider()
        print()

    input(f"  {DIM}Press Enter to return to menu...{RESET}")


# ─── Action: Exit ────────────────────────────────────────────────────────────


def action_exit() -> None:
    print()
    type_text("  Thanks for using ML Leakage Guard!", delay=0.03, color=BRIGHT_CYAN)
    type_text("  Stay safe. Prevent leakage. Save lives.", delay=0.03, color=BRIGHT_GREEN)
    print()
    farewell = rf"""
{DIM}    ╔═══════════════════════════════════════╗
    ║                                       ║
    ║   {BRIGHT_CYAN}Publication-grade ML starts here.{DIM}   ║
    ║                                       ║
    ╚═══════════════════════════════════════╝{RESET}
"""
    print(farewell)


# ─── Main Loop ────────────────────────────────────────────────────────────────


def main() -> int:
    # Handle --help
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        return 0

    while True:
        try:
            choice = show_main_menu()
        except KeyboardInterrupt:
            action_exit()
            return 0

        try:
            if choice == 1:
                action_quick_start()
            elif choice == 2:
                action_download()
            elif choice == 3:
                action_split()
            elif choice == 4:
                action_full_pipeline()
            elif choice == 5:
                action_health_check()
            elif choice == 6:
                action_guide()
            elif choice == 7:
                action_exit()
                return 0
        except KeyboardInterrupt:
            print(f"\n  {DIM}Interrupted. Returning to menu...{RESET}")
            continue

        print()
        try:
            input(f"  {DIM}Press Enter to return to menu...{RESET}")
        except (EOFError, KeyboardInterrupt):
            action_exit()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
