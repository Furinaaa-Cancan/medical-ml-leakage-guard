#!/usr/bin/env python3
"""
ML Leakage Guard -- Interactive Pipeline Wizard.

Usage:
    python3 scripts/mlgg_pixel.py
    python3 scripts/mlgg.py play
"""

from __future__ import annotations

import csv
import itertools
import locale
import os
import platform
import shutil
import subprocess
import sys
import threading
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
BG = {"b": "\033[44m", "c": "\033[46m", "g": "\033[42m", "y": "\033[43m", "k": "\033[40m"}
HIDE_CUR = "\033[?25l"
SHOW_CUR = "\033[?25h"
ERASE = "\033[2K"
UP_LINE = "\033[A"

# ── sentinel ──────────────────────────────────────────────────────────────────
BACK = type("BACK", (), {"__repr__": lambda self: "BACK"})()
SKIP = type("SKIP", (), {"__repr__": lambda self: "SKIP"})()
TOTAL_STEPS = 9

def s(fg: str, text: str, bold: bool = False) -> str:
    return f"{BOLD if bold else ''}{FG.get(fg, '')}{text}{RST}"

def _wlen(text: str) -> int:
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in text)

def _cols() -> int:
    return shutil.get_terminal_size((80, 24)).columns

def _clear() -> None:
    os.system("cls" if platform.system() == "Windows" else "clear")

def _trunc(text: str, maxw: int) -> str:
    if _wlen(text) <= maxw:
        return text
    import unicodedata
    w = 0
    for i, ch in enumerate(text):
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw + 3 > maxw:
            return text[:i] + "..."
        w += cw
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  i18n
# ══════════════════════════════════════════════════════════════════════════════

LANG = "en"

_T: Dict[str, Dict[str, str]] = {
    "lang_title":    {"en": "Language / \u8bed\u8a00", "zh": "Language / \u8bed\u8a00"},
    "lang_en":       {"en": "English", "zh": "English"},
    "lang_zh":       {"en": "\u4e2d\u6587", "zh": "\u4e2d\u6587"},

    "nav":           {"en": "[\u2191\u2193] move  [Enter] next  [q] back",
                      "zh": "[\u2191\u2193] \u79fb\u52a8  [Enter] \u4e0b\u4e00\u6b65  [q] \u8fd4\u56de"},
    "nav_first":     {"en": "[\u2191\u2193] move  [Enter] next  [q] quit",
                      "zh": "[\u2191\u2193] \u79fb\u52a8  [Enter] \u4e0b\u4e00\u6b65  [q] \u9000\u51fa"},
    "ms_hint":       {"en": "[\u2191\u2193] move  [Space] toggle  [Enter] confirm  [a] all  [q] back",
                      "zh": "[\u2191\u2193] \u79fb\u52a8  [\u7a7a\u683c] \u5207\u6362  [Enter] \u786e\u8ba4  [a] \u5168\u9009  [q] \u8fd4\u56de"},
    "bye":           {"en": "Bye!", "zh": "\u518d\u89c1\uff01"},
    "interrupted":   {"en": "Interrupted.", "zh": "\u5df2\u4e2d\u65ad\u3002"},
    "enter_continue":{"en": "Press Enter to continue...",
                      "zh": "\u6309 Enter \u7ee7\u7eed..."},

    "s_lang":        {"en": "Language", "zh": "\u8bed\u8a00"},
    "s_source":      {"en": "Data Source", "zh": "\u6570\u636e\u6765\u6e90"},
    "s_dataset":     {"en": "Dataset", "zh": "\u6570\u636e\u96c6"},
    "s_config":      {"en": "Columns", "zh": "\u5217\u914d\u7f6e"},
    "s_split":       {"en": "Split", "zh": "\u5206\u5272\u914d\u7f6e"},
    "s_models":      {"en": "Models", "zh": "\u6a21\u578b\u9009\u62e9"},
    "s_tuning":      {"en": "Tuning", "zh": "\u8c03\u4f18\u914d\u7f6e"},
    "s_confirm":     {"en": "Confirm", "zh": "\u786e\u8ba4"},
    "s_run":         {"en": "Execute", "zh": "\u6267\u884c"},

    "src_download":  {"en": "Download UCI Dataset", "zh": "\u4e0b\u8f7d UCI \u6570\u636e\u96c6"},
    "src_download_d":{"en": "Real medical datasets (heart, breast, kidney)",
                      "zh": "\u771f\u5b9e\u533b\u5b66\u6570\u636e\u96c6\uff08\u5fc3\u810f\u3001\u4e73\u817a\u3001\u80be\u75c5\uff09"},
    "src_csv":       {"en": "Use Your Own CSV", "zh": "\u4f7f\u7528\u4f60\u7684 CSV"},
    "src_csv_d":     {"en": "Bring your own dataset", "zh": "\u4f7f\u7528\u81ea\u5df1\u7684\u6570\u636e\u96c6"},
    "src_demo":      {"en": "Demo (Synthetic Data)", "zh": "\u6f14\u793a\uff08\u5408\u6210\u6570\u636e\uff09"},
    "src_demo_d":    {"en": "Auto-generated, great for first run",
                      "zh": "\u81ea\u52a8\u751f\u6210\uff0c\u9002\u5408\u9996\u6b21\u4f53\u9a8c"},

    "ds_heart":      {"en": "Heart Disease", "zh": "\u5fc3\u810f\u75c5"},
    "ds_heart_d":    {"en": "UCI Cleveland -- 297 patients, 13 features",
                      "zh": "UCI \u514b\u5229\u592b\u5170 -- 297 \u4f8b, 13 \u7279\u5f81"},
    "ds_breast":     {"en": "Breast Cancer", "zh": "\u4e73\u817a\u764c"},
    "ds_breast_d":   {"en": "UCI Wisconsin -- 569 patients, 30 features",
                      "zh": "UCI \u5a01\u65af\u5eb7\u8f9b -- 569 \u4f8b, 30 \u7279\u5f81"},
    "ds_kidney":     {"en": "Kidney Disease", "zh": "\u6162\u6027\u80be\u75c5"},
    "ds_kidney_d":   {"en": "UCI CKD -- 399 patients, 24 features",
                      "zh": "UCI CKD -- 399 \u4f8b, 24 \u7279\u5f81"},

    "pick_csv":      {"en": "Select CSV file", "zh": "\u9009\u62e9 CSV \u6587\u4ef6"},
    "manual_path":   {"en": "Enter path manually...", "zh": "\u624b\u52a8\u8f93\u5165\u8def\u5f84..."},
    "csv_prompt":    {"en": "CSV path", "zh": "CSV \u8def\u5f84"},
    "not_found":     {"en": "File not found.", "zh": "\u6587\u4ef6\u672a\u627e\u5230\u3002"},
    "bad_csv":       {"en": "Cannot read CSV header.", "zh": "\u65e0\u6cd5\u8bfb\u53d6 CSV \u8868\u5934\u3002"},

    "pick_pid":      {"en": "Patient ID column", "zh": "\u60a3\u8005 ID \u5217"},
    "pick_target":   {"en": "Target column (0/1)", "zh": "\u76ee\u6807\u53d8\u91cf\u5217 (0/1)"},
    "pick_time":     {"en": "Time / Date column", "zh": "\u65f6\u95f4\u5217"},
    "pick_strat":    {"en": "Split strategy", "zh": "\u5206\u5272\u7b56\u7565"},
    "auto":          {"en": "auto-detected", "zh": "\u81ea\u52a8\u68c0\u6d4b"},
    "no_time_col":   {"en": "No remaining columns for time.",
                      "zh": "\u6ca1\u6709\u53ef\u7528\u7684\u65f6\u95f4\u5217\u3002"},

    "strat_temporal":   {"en": "Grouped Temporal", "zh": "\u65f6\u5e8f\u5206\u7ec4"},
    "strat_temporal_d": {"en": "Sort by time, patient-disjoint (recommended)",
                         "zh": "\u6309\u65f6\u95f4\u6392\u5e8f\uff0c\u60a3\u8005\u4e0d\u76f8\u4ea4\uff08\u63a8\u8350\uff09"},
    "strat_random":     {"en": "Grouped Random", "zh": "\u968f\u673a\u5206\u7ec4"},
    "strat_random_d":   {"en": "Random patient-disjoint split",
                         "zh": "\u968f\u673a\u60a3\u8005\u4e0d\u76f8\u4ea4\u5206\u5272"},
    "strat_stratified":   {"en": "Stratified Grouped", "zh": "\u5206\u5c42\u5206\u7ec4"},
    "strat_stratified_d": {"en": "Preserve positive rate across splits",
                           "zh": "\u4fdd\u6301\u5404\u5206\u5272\u7684\u9633\u6027\u7387\u4e00\u81f4"},

    "pick_ratio":    {"en": "Train / Valid / Test ratio", "zh": "\u8bad\u7ec3 / \u9a8c\u8bc1 / \u6d4b\u8bd5 \u6bd4\u4f8b"},
    "ratio_60":      {"en": "60 / 20 / 20  (standard)", "zh": "60 / 20 / 20  \uff08\u6807\u51c6\uff09"},
    "ratio_70":      {"en": "70 / 15 / 15  (more training)", "zh": "70 / 15 / 15  \uff08\u66f4\u591a\u8bad\u7ec3\uff09"},
    "ratio_80":      {"en": "80 / 10 / 10  (small datasets)", "zh": "80 / 10 / 10  \uff08\u5c0f\u6570\u636e\u96c6\uff09"},

    "pick_models":   {"en": "Select models to train", "zh": "\u9009\u62e9\u8981\u8bad\u7ec3\u7684\u6a21\u578b"},
    "m_logistic_l1": {"en": "Logistic L1 (Lasso)", "zh": "\u903b\u8f91\u56de\u5f52 L1 (Lasso)"},
    "m_logistic_l2": {"en": "Logistic L2 (Ridge)", "zh": "\u903b\u8f91\u56de\u5f52 L2 (Ridge)"},
    "m_elasticnet":  {"en": "Logistic ElasticNet", "zh": "\u903b\u8f91\u56de\u5f52 ElasticNet"},
    "m_rf":          {"en": "Random Forest", "zh": "\u968f\u673a\u68ee\u6797"},
    "m_extra":       {"en": "Extra Trees", "zh": "\u6781\u7aef\u968f\u673a\u6811"},
    "m_hgb":         {"en": "Hist Gradient Boosting", "zh": "\u76f4\u65b9\u56fe\u68af\u5ea6\u63d0\u5347"},
    "m_ada":         {"en": "AdaBoost", "zh": "AdaBoost"},
    "m_xgb":         {"en": "XGBoost (optional)", "zh": "XGBoost\uff08\u53ef\u9009\uff09"},
    "m_cat":         {"en": "CatBoost (optional)", "zh": "CatBoost\uff08\u53ef\u9009\uff09"},

    "pick_tuning":   {"en": "Hyperparameter search", "zh": "\u8d85\u53c2\u6570\u641c\u7d22\u7b56\u7565"},
    "tune_fixed":    {"en": "Fixed Grid", "zh": "\u56fa\u5b9a\u7f51\u683c"},
    "tune_fixed_d":  {"en": "Fast, predefined hyperparameters",
                      "zh": "\u5feb\u901f\uff0c\u9884\u5b9a\u4e49\u8d85\u53c2\u6570"},
    "tune_random":   {"en": "Random Search", "zh": "\u968f\u673a\u641c\u7d22"},
    "tune_random_d": {"en": "Sample random combinations",
                      "zh": "\u968f\u673a\u91c7\u6837\u8d85\u53c2\u6570\u7ec4\u5408"},
    "tune_optuna":   {"en": "Optuna (Bayesian)", "zh": "Optuna\uff08\u8d1d\u53f6\u65af\u4f18\u5316\uff09"},
    "tune_optuna_d": {"en": "Smart search, needs optuna package",
                      "zh": "\u667a\u80fd\u641c\u7d22\uff0c\u9700\u5b89\u88c5 optuna"},

    "pick_calib":    {"en": "Probability calibration", "zh": "\u6982\u7387\u6821\u51c6\u65b9\u6cd5"},
    "calib_none":    {"en": "None", "zh": "\u65e0"},
    "calib_sig":     {"en": "Sigmoid (Platt)", "zh": "Sigmoid (Platt)"},
    "calib_iso":     {"en": "Isotonic", "zh": "Isotonic\uff08\u4fdd\u5e8f\uff09"},

    "pick_device":   {"en": "Compute device", "zh": "\u8ba1\u7b97\u8bbe\u5907"},
    "dev_auto":      {"en": "Auto", "zh": "\u81ea\u52a8"},
    "dev_auto_d":    {"en": "MPS on Mac, CUDA if available, else CPU",
                      "zh": "Mac \u7528 MPS\uff0c\u6709 CUDA \u7528 GPU\uff0c\u5426\u5219 CPU"},
    "dev_cpu":       {"en": "CPU", "zh": "CPU"},
    "dev_gpu":       {"en": "GPU / MPS", "zh": "GPU / MPS"},

    "c_file":        {"en": "File:", "zh": "\u6587\u4ef6\uff1a"},
    "c_pid":         {"en": "Patient ID:", "zh": "\u60a3\u8005 ID\uff1a"},
    "c_target":      {"en": "Target:", "zh": "\u76ee\u6807\uff1a"},
    "c_time":        {"en": "Time:", "zh": "\u65f6\u95f4\uff1a"},
    "c_strat":       {"en": "Strategy:", "zh": "\u7b56\u7565\uff1a"},
    "c_ratio":       {"en": "Ratio:", "zh": "\u6bd4\u4f8b\uff1a"},
    "c_models":      {"en": "Models:", "zh": "\u6a21\u578b\uff1a"},
    "c_tuning":      {"en": "Tuning:", "zh": "\u8c03\u4f18\uff1a"},
    "c_calib":       {"en": "Calibration:", "zh": "\u6821\u51c6\uff1a"},
    "c_device":      {"en": "Device:", "zh": "\u8bbe\u5907\uff1a"},
    "c_output":      {"en": "Output:", "zh": "\u8f93\u51fa\uff1a"},
    "c_none":        {"en": "(none)", "zh": "\uff08\u65e0\uff09"},
    "c_start":       {"en": "Start Pipeline", "zh": "\u5f00\u59cb\u8fd0\u884c"},
    "c_back":        {"en": "Go Back", "zh": "\u8fd4\u56de\u4fee\u6539"},

    "x_download":    {"en": "Downloading {ds}...", "zh": "\u6b63\u5728\u4e0b\u8f7d {ds}..."},
    "x_split":       {"en": "Splitting with safety checks...",
                      "zh": "\u6b63\u5728\u5b89\u5168\u5206\u5272..."},
    "x_train":       {"en": "Training {n} model(s)...",
                      "zh": "\u6b63\u5728\u8bad\u7ec3 {n} \u4e2a\u6a21\u578b..."},
    "x_pipeline":    {"en": "Running full pipeline...",
                      "zh": "\u6b63\u5728\u8fd0\u884c\u5b8c\u6574\u7ba1\u7ebf..."},
    "x_fail":        {"en": "Failed.", "zh": "\u5931\u8d25\u3002"},

    "r_done":        {"en": "Complete!", "zh": "\u5b8c\u6210\uff01"},
    "r_split_ok":    {"en": "Split Complete!", "zh": "\u5206\u5272\u5b8c\u6210\uff01"},
    "r_train_ok":    {"en": "Training Complete!", "zh": "\u8bad\u7ec3\u5b8c\u6210\uff01"},
    "r_saved":       {"en": "Saved to:", "zh": "\u4fdd\u5b58\u81f3\uff1a"},
    "r_next":        {"en": "All done! Results saved to output directory.",
                      "zh": "\u5168\u90e8\u5b8c\u6210\uff01\u7ed3\u679c\u5df2\u4fdd\u5b58\u81f3\u8f93\u51fa\u76ee\u5f55\u3002"},

    "rows":          {"en": "rows", "zh": "\u884c"},
    "patients":      {"en": "patients", "zh": "\u60a3\u8005"},
    "columns":       {"en": "columns", "zh": "\u5217"},
}


def t(key: str, **kwargs: Any) -> str:
    val = _T.get(key, {}).get(LANG, _T.get(key, {}).get("en", key))
    if kwargs:
        val = val.format(**kwargs)
    return val


def detect_lang() -> str:
    try:
        loc = locale.getlocale()[0] or os.environ.get("LANG", "")
        if loc.startswith("zh"):
            return "zh"
    except Exception:
        pass
    return "en"


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL INPUT
# ══════════════════════════════════════════════════════════════════════════════

def _getch() -> str:
    """Read one keypress using os.read (unbuffered) to avoid Python stdin buffering."""
    try:
        import tty, termios, select as sel_mod
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = os.read(fd, 1)
            if ch == b"\x1b":
                if not sel_mod.select([fd], [], [], 0.05)[0]:
                    return "ESC"
                ch2 = os.read(fd, 1)
                if ch2 == b"[":
                    ch3 = os.read(fd, 1)
                    if ch3 == b"A": return "UP"
                    if ch3 == b"B": return "DOWN"
                    while sel_mod.select([fd], [], [], 0.02)[0]:
                        os.read(fd, 1)
                return "ESC"
            if ch in (b"\r", b"\n"): return "ENTER"
            if ch == b"\x03": return "CTRL_C"
            if ch == b"\x04": return "CTRL_D"
            if ch == b" ": return "SPACE"
            if ch == b"q": return "Q"
            if ch == b"a": return "A"
            return ch.decode("latin-1")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        raw = input()
        return raw.strip() or "ENTER"


# ══════════════════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def step_header(step: int, total: int, title: str) -> None:
    w = min(_cols() - 4, 64)
    bar_w = max(w - 28, 8)
    filled = int(bar_w * step / total)
    bar = s('C', '\u2593' * filled) + DIM + '\u2591' * (bar_w - filled) + RST
    pct = f"{int(100 * step / total):>3}%"
    print(f"  {s('C', 'LEAKAGE GUARD', bold=True)}  {DIM}Step {step}/{total}{RST}  {bar} {DIM}{pct}{RST}")
    print(f"  {s('C', '\u2500' * w)}")
    if title:
        print(f"\n  {s('W', title, bold=True)}\n")


def select(options: List[str], descs: Optional[List[str]] = None,
           title: str = "", is_first: bool = False) -> int:
    sel = 0
    n = len(options)
    if n == 0:
        return -1
    has_desc = descs and len(descs) == n
    maxw = _cols() - 10

    def _draw() -> None:
        sys.stdout.write(HIDE_CUR); sys.stdout.flush()
        if title:
            print(f"  {s('C', title, bold=True)}")
            print()
        for i in range(n):
            lbl = _trunc(options[i], maxw - 4)
            if i == sel:
                line = f"  {s('C','>', bold=True)} {BG['b']}{FG['W']}{BOLD} {lbl} {RST}"
                desc_room = maxw - _wlen(lbl) - 8
                if has_desc and descs[i] and desc_room > 8:
                    d = _trunc(descs[i], desc_room)
                    line += f"  {s('C', d)}"
                print(line)
            else:
                line = f"    {DIM}{lbl}{RST}"
                desc_room = maxw - _wlen(lbl) - 8
                if has_desc and descs[i] and desc_room > 8:
                    d = _trunc(descs[i], desc_room)
                    line += f"  {DIM}{d}{RST}"
                print(line)
        print()
        hint = t("nav_first") if is_first else t("nav")
        print(f"  {DIM}{hint}{RST}")

    title_lines = 2 if title else 0
    line_count = n + title_lines + 2

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
        elif len(key) == 1 and key.isdigit() and 1 <= int(key) <= min(n, 9):
            sel = int(key) - 1
        else:
            continue
        for _ in range(line_count):
            sys.stdout.write(f"{UP_LINE}{ERASE}")
        sys.stdout.write("\r"); sys.stdout.flush()
        _draw()


def multi_select(options: List[str], descs: Optional[List[str]] = None,
                 title: str = "", defaults: Optional[List[int]] = None) -> Optional[List[int]]:
    """Multi-select with checkboxes. Returns list of indices or None on quit."""
    sel = 0
    n = len(options)
    if n == 0:
        return None
    has_desc = descs and len(descs) == n
    checked: set = set(defaults or [])
    maxw = _cols() - 10

    def _draw() -> None:
        sys.stdout.write(HIDE_CUR); sys.stdout.flush()
        if title:
            print(f"  {s('C', title, bold=True)}")
            print()
        for i in range(n):
            mark = s('G', '\u2713') if i in checked else ' '
            lbl = _trunc(options[i], maxw - 10)
            if i == sel:
                line = f"  {s('C','>', bold=True)} [{mark}] {BG['b']}{FG['W']}{BOLD}{lbl}{RST}"
                desc_room = maxw - _wlen(options[i]) - 14
                if has_desc and descs[i] and desc_room > 8:
                    d = _trunc(descs[i], desc_room)
                    line += f"  {s('C', d)}"
                print(line)
            else:
                line = f"    [{mark}] {DIM}{lbl}{RST}"
                desc_room = maxw - _wlen(options[i]) - 14
                if has_desc and descs[i] and desc_room > 8:
                    d = _trunc(descs[i], desc_room)
                    line += f"  {DIM}{d}{RST}"
                print(line)
        print()
        print(f"  {DIM}{t('ms_hint')}{RST}")

    title_lines = 2 if title else 0
    line_count = n + title_lines + 2

    _draw()
    while True:
        key = _getch()
        if key == "UP" and sel > 0:
            sel -= 1
        elif key == "DOWN" and sel < n - 1:
            sel += 1
        elif key == "SPACE":
            if sel in checked:
                checked.discard(sel)
            else:
                checked.add(sel)
        elif key == "A":
            if len(checked) == n:
                checked.clear()
            else:
                checked = set(range(n))
        elif key == "ENTER":
            sys.stdout.write(SHOW_CUR); sys.stdout.flush()
            return sorted(checked)
        elif key in ("Q", "CTRL_C", "CTRL_D", "ESC"):
            sys.stdout.write(SHOW_CUR); sys.stdout.flush()
            return None
        else:
            continue
        for _ in range(line_count):
            sys.stdout.write(f"{UP_LINE}{ERASE}")
        sys.stdout.write("\r"); sys.stdout.flush()
        _draw()


def box(title: str, lines: List[str], color: str = "C") -> None:
    maxw = _cols() - 6
    w = min(max(max((_wlen(l) for l in lines), default=0), _wlen(title)) + 4, maxw)
    print(f"  {s(color, '\u250c' + '\u2500' * w + '\u2510')}")
    if title:
        pad = w - _wlen(title) - 2
        print(f"  {s(color, '\u2502')} {s('W', title, bold=True)}{' ' * max(pad,0)}{s(color, '\u2502')}")
        print(f"  {s(color, '\u251c' + '\u2500' * w + '\u2524')}")
    for line in lines:
        tl = _trunc(line, w - 2)
        pad = w - _wlen(tl) - 2
        print(f"  {s(color, '\u2502')} {tl}{' ' * max(pad, 0)}{s(color, '\u2502')}")
    print(f"  {s(color, '\u2514' + '\u2500' * w + '\u2518')}")


class Spinner:
    FRAMES = ["\u28cb", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
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
        for f in itertools.cycle(self.FRAMES):
            if self._stop.is_set(): break
            sys.stdout.write(f"\r  {s('C', f)} {s('W', self.label)}")
            sys.stdout.flush()
            self._stop.wait(0.08)


def run_spinner(cmd: List[str], label: str, cwd: str = "") -> Tuple[int, str, str]:
    with Spinner(label):
        p = subprocess.run(cmd, cwd=cwd or str(REPO_ROOT),
                           capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


# ══════════════════════════════════════════════════════════════════════════════
#  SMART DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_PID_HINTS = ["patient_id", "patientid", "patient", "subject_id", "subjectid",
              "id", "pid", "mrn", "record_id", "case_id"]
_TGT_HINTS = ["target", "label", "y", "outcome", "diagnosis", "class",
              "result", "status", "disease", "mortality"]
_TIME_HINTS = ["time", "date", "timestamp", "event_time", "datetime",
               "admission", "visit_date", "created_at"]


def detect_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    pid = target = time_col = None
    for col in cols:
        low = col.lower().strip().replace(" ", "_")
        if not pid:
            for h in _PID_HINTS:
                if h in low: pid = col; break
        if not target:
            for h in _TGT_HINTS:
                if h in low: target = col; break
        if not time_col:
            for h in _TIME_HINTS:
                if h in low: time_col = col; break
    return {"pid": pid, "target": target, "time": time_col}


def scan_csv() -> List[Path]:
    found: List[Path] = []
    for d in [EXAMPLES_DIR, DESKTOP, Path.home()/"Downloads",
              Path.home()/"Documents", REPO_ROOT, DEFAULT_OUT]:
        if d.is_dir():
            try:
                for f in sorted(d.glob("*.csv"))[:10]:
                    try:
                        if f not in found and f.stat().st_size > 100:
                            found.append(f)
                    except (PermissionError, OSError):
                        pass
            except (PermissionError, OSError):
                pass
    return found[:15]


def csv_cols(path: Path) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return [c.strip() for c in next(csv.reader(fh), []) if c.strip()]
    except Exception:
        return []


def csv_rows(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh) - 1
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
#  LOGO
# ══════════════════════════════════════════════════════════════════════════════

LOGO = f"""
{s('C','',True)}
    \u2588\u2588\u2557     \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557  \u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557
    \u2588\u2588\u2551     \u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551 \u2588\u2588\u2554\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d \u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d
    \u2588\u2588\u2551     \u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2554\u255d \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2557
    \u2588\u2588\u2551     \u2588\u2588\u2554\u2550\u2550\u255d  \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2588\u2588\u2557 \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u255d
    \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2557\u2588\u2588\u2551  \u2588\u2588\u2551\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557
    \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d{RST}
{s('Y','',True)}
     \u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557   \u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2557
    \u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d \u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557
    \u2588\u2588\u2551  \u2588\u2588\u2588\u2557\u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551  \u2588\u2588\u2551
    \u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551  \u2588\u2588\u2551
    \u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d
     \u255a\u2550\u2550\u2550\u2550\u2550\u255d  \u255a\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u255d {RST}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL POOL
# ══════════════════════════════════════════════════════════════════════════════

MODEL_POOL = [
    ("logistic_l1",               "m_logistic_l1"),
    ("logistic_l2",               "m_logistic_l2"),
    ("logistic_elasticnet",       "m_elasticnet"),
    ("random_forest_balanced",    "m_rf"),
    ("extra_trees_balanced",      "m_extra"),
    ("hist_gradient_boosting_l2", "m_hgb"),
    ("adaboost",                  "m_ada"),
    ("xgboost",                   "m_xgb"),
    ("catboost",                  "m_cat"),
]
DEFAULT_MODELS = [0, 1, 3, 5]  # L1, L2, RF, HGB


# ══════════════════════════════════════════════════════════════════════════════
#  WIZARD STEPS
# ══════════════════════════════════════════════════════════════════════════════

def step_lang(state: Dict) -> Any:
    _clear()
    print(LOGO)
    step_header(1, TOTAL_STEPS, "")
    ci = select([t("lang_en"), t("lang_zh")], title=t("lang_title"), is_first=True)
    if ci < 0:
        return BACK
    global LANG
    LANG = "en" if ci == 0 else "zh"
    state["lang"] = LANG
    return True


def step_source(state: Dict) -> Any:
    _clear()
    step_header(2, TOTAL_STEPS, t("s_source"))
    ci = select(
        [t("src_download"), t("src_csv"), t("src_demo")],
        [t("src_download_d"), t("src_csv_d"), t("src_demo_d")],
    )
    if ci < 0:
        return BACK
    state["source"] = ["download", "csv", "demo"][ci]
    return True


def step_dataset(state: Dict) -> Any:
    source = state["source"]

    if source == "demo":
        state["csv_path"] = None
        state["dataset_key"] = "demo"
        state["pid"] = "patient_id"
        state["target"] = "y"
        state["time"] = "event_time"
        state["strategy"] = "grouped_temporal"
        state["train_ratio"] = 0.6
        state["valid_ratio"] = 0.2
        state["test_ratio"] = 0.2
        state["model_pool"] = ""
        state["hyperparam_search"] = "fixed_grid"
        state["calibration"] = "none"
        state["device"] = "auto"
        state["out_dir"] = str(DEFAULT_OUT / "pipeline")
        return SKIP

    _clear()
    step_header(3, TOTAL_STEPS, t("s_dataset"))

    if source == "download":
        ci = select(
            [t("ds_heart"), t("ds_breast"), t("ds_kidney")],
            [t("ds_heart_d"), t("ds_breast_d"), t("ds_kidney_d")],
        )
        if ci < 0:
            return BACK
        keys = ["heart", "breast", "ckd"]
        files = ["heart_disease", "breast_cancer", "chronic_kidney_disease"]
        state["dataset_key"] = keys[ci]
        state["dataset_file"] = files[ci]
        state["csv_path"] = str(EXAMPLES_DIR / f"{files[ci]}.csv")
        state["out_dir"] = str(DEFAULT_OUT / files[ci])
        state["pid"] = "patient_id"
        state["target"] = "y"
        state["time"] = "event_time"
        return True

    # source == "csv"
    files = scan_csv()
    if files:
        names = [f.name for f in files]
        descs = [f"{csv_rows(f)} {t('rows')}  --  {f.parent}" for f in files]
        names.append(t("manual_path"))
        descs.append("")
        fi = select(names, descs, title=t("pick_csv"))
        if fi < 0:
            return BACK
        if fi == len(files):
            sys.stdout.write(SHOW_CUR)
            path = input(f"  {s('C','>')} {s('W', t('csv_prompt'))}: ").strip()
        else:
            path = str(files[fi])
    else:
        sys.stdout.write(SHOW_CUR)
        path = input(f"  {s('C','>')} {s('W', t('csv_prompt'))}: ").strip()

    if not path or not Path(path).exists():
        print(f"\n  {s('R', t('not_found'))}")
        sys.stdout.write(SHOW_CUR)
        try:
            input(f"  {DIM}{t('enter_continue')}{RST}")
        except (EOFError, KeyboardInterrupt):
            pass
        return BACK

    state["csv_path"] = path
    state["dataset_key"] = "custom"
    state["out_dir"] = str(DEFAULT_OUT / Path(path).stem)
    return True


def step_config(state: Dict) -> Any:
    """Column selection -- Patient ID + Target (CSV mode only)."""
    if state.get("source") in ("demo", "download"):
        return SKIP

    csv_path = state["csv_path"]
    columns = csv_cols(Path(csv_path))
    if not columns:
        _clear()
        step_header(4, TOTAL_STEPS, t("s_config"))
        print(f"  {s('R', t('bad_csv'))}")
        sys.stdout.write(SHOW_CUR)
        try:
            input(f"  {DIM}{t('enter_continue')}{RST}")
        except (EOFError, KeyboardInterrupt):
            pass
        return BACK

    detected = detect_columns(columns)
    rows = csv_rows(Path(csv_path))
    state["_columns"] = columns
    state["_detected"] = detected
    state["_rows"] = rows

    def _config_header(*chosen: Tuple[str, str]) -> None:
        _clear()
        step_header(4, TOTAL_STEPS, t("s_config"))
        print(f"  {s('W', Path(csv_path).name, bold=True)}  {DIM}{rows} {t('rows')}, {len(columns)} {t('columns')}{RST}")
        if not chosen and any(detected.values()):
            hints = []
            if detected["pid"]: hints.append(f"ID={detected['pid']}")
            if detected["target"]: hints.append(f"Target={detected['target']}")
            if detected["time"]: hints.append(f"Time={detected['time']}")
            auto_label = t('auto')
            print(f"  {s('G', '[' + auto_label + ']')} {DIM}{', '.join(hints)}{RST}")
        for label, value in chosen:
            print(f"  {s('G', '\u2713')} {label} {s('W', value)}")
        print()

    # Patient ID
    _config_header()
    pid_opts = columns[:]
    if detected["pid"] and detected["pid"] in pid_opts:
        pid_opts.remove(detected["pid"])
        pid_opts.insert(0, detected["pid"])
    pi = select(pid_opts, title=t("pick_pid"))
    if pi < 0: return BACK
    pid = pid_opts[pi]

    # Target
    _config_header((t("c_pid"), pid))
    rem1 = [c for c in columns if c != pid]
    tgt_opts = rem1[:]
    if detected["target"] and detected["target"] in tgt_opts:
        tgt_opts.remove(detected["target"])
        tgt_opts.insert(0, detected["target"])
    ti = select(tgt_opts, title=t("pick_target"))
    if ti < 0: return BACK
    tgt = tgt_opts[ti]

    state["pid"] = pid
    state["target"] = tgt
    return True


def step_split(state: Dict) -> Any:
    """Strategy + time column (if temporal) + split ratio."""
    if state.get("source") == "demo":
        return SKIP

    source = state["source"]
    _clear()
    step_header(5, TOTAL_STEPS, t("s_split"))

    # Strategy
    si = select(
        [t("strat_temporal"), t("strat_random"), t("strat_stratified")],
        [t("strat_temporal_d"), t("strat_random_d"), t("strat_stratified_d")],
        title=t("pick_strat"),
    )
    if si < 0: return BACK
    strat = ["grouped_temporal", "grouped_random", "stratified_grouped"][si]
    state["strategy"] = strat

    # Time column (if temporal strategy)
    tcol = ""
    if strat == "grouped_temporal":
        if source == "csv":
            columns = state.get("_columns", [])
            pid = state.get("pid", "")
            tgt = state.get("target", "")
            detected = state.get("_detected", {})
            rem = [c for c in columns if c not in (pid, tgt)]
            if not rem:
                print(f"\n  {s('R', t('no_time_col'))}")
                sys.stdout.write(SHOW_CUR)
                try:
                    input(f"  {DIM}{t('enter_continue')}{RST}")
                except (EOFError, KeyboardInterrupt):
                    pass
                return BACK
            _clear()
            step_header(5, TOTAL_STEPS, t("s_split"))
            print(f"  {s('G', '\u2713')} {t('c_strat')} {s('W', strat)}\n")
            time_opts = rem[:]
            if detected.get("time") and detected["time"] in time_opts:
                time_opts.remove(detected["time"])
                time_opts.insert(0, detected["time"])
            tci = select(time_opts, title=t("pick_time"))
            if tci < 0: return BACK
            tcol = time_opts[tci]
        else:
            tcol = state.get("time", "event_time")
    state["time"] = tcol

    # Ratio
    _clear()
    step_header(5, TOTAL_STEPS, t("s_split"))
    print(f"  {s('G', '\u2713')} {t('c_strat')} {s('W', strat)}")
    if tcol:
        print(f"  {s('G', '\u2713')} {t('c_time')} {s('W', tcol)}")
    print()
    ri = select(
        [t("ratio_60"), t("ratio_70"), t("ratio_80")],
        title=t("pick_ratio"),
    )
    if ri < 0: return BACK
    ratios = [(0.6, 0.2, 0.2), (0.7, 0.15, 0.15), (0.8, 0.1, 0.1)]
    state["train_ratio"], state["valid_ratio"], state["test_ratio"] = ratios[ri]
    return True


def step_models(state: Dict) -> Any:
    """Multi-select model families."""
    if state.get("source") == "demo":
        return SKIP

    _clear()
    step_header(6, TOTAL_STEPS, t("s_models"))

    labels = [t(key) for _, key in MODEL_POOL]
    selected = multi_select(labels, title=t("pick_models"), defaults=DEFAULT_MODELS)
    if selected is None:
        return BACK
    if not selected:
        selected = list(DEFAULT_MODELS)
    state["model_pool"] = ",".join(MODEL_POOL[i][0] for i in selected)
    state["_model_labels"] = [labels[i] for i in selected]
    return True


def step_tuning(state: Dict) -> Any:
    """Tuning strategy + calibration + device."""
    if state.get("source") == "demo":
        return SKIP

    _clear()
    step_header(7, TOTAL_STEPS, t("s_tuning"))

    # Tuning strategy
    ti = select(
        [t("tune_fixed"), t("tune_random"), t("tune_optuna")],
        [t("tune_fixed_d"), t("tune_random_d"), t("tune_optuna_d")],
        title=t("pick_tuning"),
    )
    if ti < 0: return BACK
    state["hyperparam_search"] = ["fixed_grid", "random_subsample", "optuna"][ti]

    # Calibration
    _clear()
    step_header(7, TOTAL_STEPS, t("s_tuning"))
    print(f"  {s('G', '\u2713')} {t('c_tuning')} {s('W', state['hyperparam_search'])}\n")
    ci = select(
        [t("calib_none"), t("calib_sig"), t("calib_iso")],
        title=t("pick_calib"),
    )
    if ci < 0: return BACK
    state["calibration"] = ["none", "sigmoid", "isotonic"][ci]

    # Device
    _clear()
    step_header(7, TOTAL_STEPS, t("s_tuning"))
    print(f"  {s('G', '\u2713')} {t('c_tuning')} {s('W', state['hyperparam_search'])}")
    print(f"  {s('G', '\u2713')} {t('c_calib')} {s('W', state['calibration'])}\n")
    di = select(
        [t("dev_auto"), t("dev_cpu"), t("dev_gpu")],
        [t("dev_auto_d"), "", ""],
        title=t("pick_device"),
    )
    if di < 0: return BACK
    state["device"] = ["auto", "cpu", "gpu"][di]
    return True


def step_confirm(state: Dict) -> Any:
    _clear()
    step_header(8, TOTAL_STEPS, t("s_confirm"))

    if state["dataset_key"] == "demo":
        box("Demo Pipeline", [
            f"{t('c_output')} {state['out_dir']}/",
            "",
            "Synthetic data | 28 safety gates | ~5 min",
        ], color="C")
    else:
        fname = Path(state["csv_path"]).name if state.get("csv_path") else "?"
        ratio_str = f"{int(state.get('train_ratio',0.6)*100)}/{int(state.get('valid_ratio',0.2)*100)}/{int(state.get('test_ratio',0.2)*100)}"
        models_str = ", ".join(state.get("_model_labels", ["?"]))

        all_labels = [t('c_file'), t('c_pid'), t('c_target'), t('c_time'),
                      t('c_strat'), t('c_ratio'), t('c_models'), t('c_tuning'),
                      t('c_calib'), t('c_device'), t('c_output')]
        col_w = max(_wlen(l) for l in all_labels) + 2
        def _p(label: str) -> str:
            return label + " " * max(col_w - _wlen(label), 1)

        lines = [
            f"{_p(t('c_file'))}{fname}",
            f"{_p(t('c_pid'))}{state.get('pid', '?')}",
            f"{_p(t('c_target'))}{state.get('target', '?')}",
            f"{_p(t('c_time'))}{state.get('time') or t('c_none')}",
            f"{_p(t('c_strat'))}{state.get('strategy', '?')}",
            f"{_p(t('c_ratio'))}{ratio_str}",
            "",
            f"{_p(t('c_models'))}{models_str}",
            f"{_p(t('c_tuning'))}{state.get('hyperparam_search', '?')}",
            f"{_p(t('c_calib'))}{state.get('calibration', '?')}",
            f"{_p(t('c_device'))}{state.get('device', '?')}",
            "",
            f"{_p(t('c_output'))}{state['out_dir']}/",
        ]
        box(t("s_confirm"), lines, color="C")

    print()
    ci = select([t("c_start"), t("c_back")])
    if ci != 0:
        return BACK
    return True


def step_run(state: Dict) -> Any:
    _clear()
    step_header(9, TOTAL_STEPS, t("s_run"))

    source = state["source"]

    # ── Demo: run full onboarding pipeline ──
    if source == "demo":
        rc, _, err = run_spinner(
            [sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
             "--project-root", state["out_dir"], "--mode", "guided", "--yes"],
            t("x_pipeline"),
        )
        print()
        if rc == 0:
            box(t("r_done"), [
                f"{t('c_output')} {state['out_dir']}/",
                "  evidence/  -- audit artifacts",
                "  models/    -- trained model",
                "  data/      -- split datasets",
            ], color="G")
        else:
            print(f"  {s('R', t('x_fail'))}")
            if err:
                for l in err.strip().split("\n")[-5:]:
                    print(f"  {DIM}{l}{RST}")
        return True

    # ── Download + Split + Train flow ──
    completed: List[Tuple[str, str]] = []  # (label, "done"|"fail")
    out_data = str(Path(state["out_dir"]) / "data")

    def _progress() -> None:
        for label, st in completed:
            icon = s('G', '[ok]') if st == "done" else s('R', '[!!]')
            clr = 'W' if st == "done" else 'R'
            print(f"  {icon}  {s(clr, label)}")

    # ── Phase 1: Download ──
    if source == "download":
        ds_names = {"heart": t("ds_heart"), "breast": t("ds_breast"), "ckd": t("ds_kidney")}
        ds_label = ds_names.get(state.get("dataset_key", ""), state.get("dataset_key", ""))
        dl_label = t("x_download", ds=ds_label)

        rc, _, err = run_spinner(
            [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"),
             state["dataset_key"]],
            dl_label,
        )
        if rc != 0:
            completed.append((dl_label, "fail"))
            _clear(); step_header(9, TOTAL_STEPS, t("s_run")); _progress()
            print(f"\n  {s('R', t('x_fail'))}")
            if err:
                for l in err.strip().split("\n")[-3:]:
                    print(f"  {DIM}{l}{RST}")
            return True
        completed.append((dl_label, "done"))

    # ── Phase 2: Split ──
    _clear(); step_header(9, TOTAL_STEPS, t("s_run")); _progress()
    split_label = t("x_split")
    csv_path = state["csv_path"]

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
        "--input", csv_path, "--output-dir", out_data,
        "--patient-id-col", state["pid"], "--target-col", state["target"],
        "--strategy", state["strategy"],
        "--train-ratio", str(state["train_ratio"]),
        "--valid-ratio", str(state["valid_ratio"]),
        "--test-ratio", str(state["test_ratio"]),
    ]
    if state.get("time"):
        cmd.extend(["--time-col", state["time"]])

    rc, _, err = run_spinner(cmd, split_label)
    if rc != 0:
        completed.append((split_label, "fail"))
        _clear(); step_header(9, TOTAL_STEPS, t("s_run")); _progress()
        print(f"\n  {s('R', t('x_fail'))}")
        if err:
            for l in err.strip().split("\n")[-3:]:
                print(f"  {DIM}{l}{RST}")
        return True
    completed.append((split_label, "done"))

    # Show split results
    _clear(); step_header(9, TOTAL_STEPS, t("s_run")); _progress()
    try:
        import pandas as pd
        tr = pd.read_csv(Path(out_data) / "train.csv")
        va = pd.read_csv(Path(out_data) / "valid.csv")
        te = pd.read_csv(Path(out_data) / "test.csv")
        pid_col = state["pid"]
        print()
        box(t("r_split_ok"), [
            f"train.csv  {len(tr):>5} {t('rows')}  {tr[pid_col].nunique():>4} {t('patients')}",
            f"valid.csv  {len(va):>5} {t('rows')}  {va[pid_col].nunique():>4} {t('patients')}",
            f"test.csv   {len(te):>5} {t('rows')}  {te[pid_col].nunique():>4} {t('patients')}",
        ], color="G")
    except Exception:
        print(f"\n  {s('G', '[ok]', True)} {out_data}/")
    print()

    # ── Phase 3: Train ──
    model_count = len(state.get("model_pool", "").split(","))
    train_label = t("x_train", n=model_count)

    evidence_dir = str(Path(state["out_dir"]) / "evidence")
    models_dir = str(Path(state["out_dir"]) / "models")
    Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    ignore_parts = [state["pid"]]
    if state.get("time"):
        ignore_parts.append(state["time"])
    if source == "download" and "event_time" not in ignore_parts:
        ignore_parts.append("event_time")
    ignore_cols = ",".join(ignore_parts)

    train_cmd = [
        sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "train", "--",
        "--train", str(Path(out_data) / "train.csv"),
        "--valid", str(Path(out_data) / "valid.csv"),
        "--test", str(Path(out_data) / "test.csv"),
        "--target-col", state["target"],
        "--patient-id-col", state["pid"],
        "--ignore-cols", ignore_cols,
        "--model-pool", state["model_pool"],
        "--hyperparam-search", state["hyperparam_search"],
        "--calibration-method", state["calibration"],
        "--device", state["device"],
        "--model-selection-report-out", str(Path(evidence_dir) / "model_selection_report.json"),
        "--evaluation-report-out", str(Path(evidence_dir) / "evaluation_report.json"),
        "--model-out", str(Path(models_dir) / "model.pkl"),
    ]

    rc, _, err = run_spinner(train_cmd, train_label)
    if rc != 0:
        completed.append((train_label, "fail"))
        _clear(); step_header(9, TOTAL_STEPS, t("s_run")); _progress()
        print(f"\n  {s('R', t('x_fail'))}")
        if err:
            for l in err.strip().split("\n")[-5:]:
                print(f"  {DIM}{l}{RST}")
        return True
    completed.append((train_label, "done"))

    # Show final results
    _clear(); step_header(9, TOTAL_STEPS, t("s_run")); _progress()
    print()
    box(t("r_train_ok"), [
        f"{t('c_output')} {state['out_dir']}/",
        "  evidence/  -- model_selection_report, evaluation_report",
        "  models/    -- trained model artifact",
        "  data/      -- train / valid / test splits",
    ], color="G")

    print(f"\n  {DIM}{t('r_next')}{RST}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WIZARD
# ══════════════════════════════════════════════════════════════════════════════

def wizard() -> int:
    global LANG
    LANG = detect_lang()

    state: Dict[str, Any] = {}
    steps = [step_lang, step_source, step_dataset, step_config,
             step_split, step_models, step_tuning,
             step_confirm, step_run]
    skipped: set = set()
    i = 0

    while i < len(steps):
        try:
            result = steps[i](state)
        except KeyboardInterrupt:
            print(f"\n  {DIM}{t('interrupted')}{RST}")
            return 0

        if result is BACK:
            if i == 0:
                print(f"\n  {s('C', t('bye'))}\n")
                return 0
            i -= 1
            while i > 0 and i in skipped:
                i -= 1
        elif result is SKIP:
            skipped.add(i)
            i += 1
        else:
            skipped.discard(i)
            i += 1

    print()
    sys.stdout.write(SHOW_CUR); sys.stdout.flush()
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass
    try:
        input(f"  {DIM}{t('enter_continue')}{RST}")
    except (EOFError, KeyboardInterrupt):
        pass
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        return 0
    return wizard()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        sys.stdout.write(SHOW_CUR); sys.stdout.flush()
