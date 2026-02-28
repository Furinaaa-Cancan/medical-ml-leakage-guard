#!/usr/bin/env python3
"""
ML Leakage Guard -- Pixel-Art Interactive CLI.

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
MOUSE_ON = "\033[?1000h\033[?1006h"   # enable SGR mouse tracking
MOUSE_OFF = "\033[?1000l\033[?1006l"  # disable mouse tracking
GET_CURSOR = "\033[6n"                 # request cursor position

def s(fg: str, text: str, bold: bool = False) -> str:
    return f"{BOLD if bold else ''}{FG.get(fg,'')}{text}{RST}"

def _wlen(text: str) -> int:
    """Display width of text -- CJK chars count as 2 columns."""
    import unicodedata
    w = 0
    for ch in text:
        cat = unicodedata.east_asian_width(ch)
        w += 2 if cat in ("W", "F") else 1
    return w

def _clear() -> None:
    os.system("cls" if platform.system() == "Windows" else "clear")


# ══════════════════════════════════════════════════════════════════════════════
#  i18n  --  All UI text in English and Chinese
# ══════════════════════════════════════════════════════════════════════════════

LANG = "en"  # global, set at startup

_T: Dict[str, Dict[str, str]] = {
    # ── language picker ───────────────────────────────────────────────────
    "lang_title":       {"en": "Language / \u8bed\u8a00", "zh": "Language / \u8bed\u8a00"},
    "lang_en":          {"en": "English", "zh": "English"},
    "lang_zh":          {"en": "\u4e2d\u6587", "zh": "\u4e2d\u6587"},

    # ── nav hints ─────────────────────────────────────────────────────────
    "nav_hint":         {"en": "[Up/Down] move  [Enter] select  [Click] pick  [q] back",
                         "zh": "[\u4e0a/\u4e0b] \u79fb\u52a8  [Enter] \u786e\u8ba4  [\u70b9\u51fb] \u9009\u62e9  [q] \u8fd4\u56de"},
    "press_enter":      {"en": "Press Enter to return to menu...",
                         "zh": "\u6309 Enter \u8fd4\u56de\u4e3b\u83dc\u5355..."},
    "bye":              {"en": "Bye!", "zh": "\u518d\u89c1\uff01"},
    "interrupted":      {"en": "Interrupted.", "zh": "\u5df2\u4e2d\u65ad\u3002"},

    # ── home menu ─────────────────────────────────────────────────────────
    "subtitle":         {"en": "Medical ML Data Leakage Prevention Pipeline",
                         "zh": "\u533b\u5b66 ML \u6570\u636e\u6cc4\u6f0f\u9632\u62a4\u7ba1\u7ebf"},
    "tagline":          {"en": "28 Fail-Closed Gates | Publication-Grade Evidence",
                         "zh": "28 \u4e2a\u5931\u8d25\u5173\u95ed\u5b89\u5168\u95e8 | \u53d1\u8868\u7ea7\u8bc1\u636e"},
    "m_quick":          {"en": "Quick Start", "zh": "\u5feb\u901f\u5f00\u59cb"},
    "m_quick_d":        {"en": "One-click: download + split a real dataset",
                         "zh": "\u4e00\u952e\u4e0b\u8f7d + \u5206\u5272\u771f\u5b9e\u6570\u636e\u96c6"},
    "m_download":       {"en": "Download", "zh": "\u4e0b\u8f7d\u6570\u636e\u96c6"},
    "m_download_d":     {"en": "Get UCI medical datasets",
                         "zh": "\u83b7\u53d6 UCI \u533b\u5b66\u6570\u636e\u96c6"},
    "m_split":          {"en": "Split CSV", "zh": "\u5206\u5272 CSV"},
    "m_split_d":        {"en": "Split your own CSV with safety guarantees",
                         "zh": "\u4ee5\u533b\u5b66\u5b89\u5168\u4fdd\u8bc1\u5206\u5272\u4f60\u7684 CSV"},
    "m_pipeline":       {"en": "Full Pipeline", "zh": "\u5b8c\u6574\u7ba1\u7ebf"},
    "m_pipeline_d":     {"en": "End-to-end training with 28 gates",
                         "zh": "28 \u4e2a\u5b89\u5168\u95e8\u7aef\u5230\u7aef\u8bad\u7ec3"},
    "m_health":         {"en": "Health Check", "zh": "\u73af\u5883\u68c0\u67e5"},
    "m_health_d":       {"en": "Verify Python, packages, CLI",
                         "zh": "\u68c0\u67e5 Python\u3001\u4f9d\u8d56\u5305\u3001CLI"},
    "m_guide":          {"en": "Guide", "zh": "\u4f7f\u7528\u6307\u5357"},
    "m_guide_d":        {"en": "Learn about data leakage prevention",
                         "zh": "\u4e86\u89e3\u6570\u636e\u6cc4\u6f0f\u9632\u62a4"},
    "m_lang":           {"en": "Switch Language", "zh": "\u5207\u6362\u8bed\u8a00"},
    "m_lang_d":         {"en": "Currently: English",
                         "zh": "\u5f53\u524d\uff1a\u4e2d\u6587"},
    "m_quit":           {"en": "Quit", "zh": "\u9000\u51fa"},
    "m_quit_d":         {"en": "", "zh": ""},

    # ── datasets ──────────────────────────────────────────────────────────
    "ds_heart":         {"en": "Heart Disease", "zh": "\u5fc3\u810f\u75c5"},
    "ds_heart_d":       {"en": "UCI Cleveland -- 297 patients, 13 features",
                         "zh": "UCI \u514b\u5229\u592b\u5170 -- 297 \u4f8b, 13 \u7279\u5f81"},
    "ds_breast":        {"en": "Breast Cancer", "zh": "\u4e73\u817a\u764c"},
    "ds_breast_d":      {"en": "UCI Wisconsin -- 569 patients, 30 features",
                         "zh": "UCI \u5a01\u65af\u5eb7\u8f9b -- 569 \u4f8b, 30 \u7279\u5f81"},
    "ds_kidney":        {"en": "Kidney Disease", "zh": "\u6162\u6027\u80be\u75c5"},
    "ds_kidney_d":      {"en": "UCI CKD -- 399 patients, 24 features",
                         "zh": "UCI CKD -- 399 \u4f8b, 24 \u7279\u5f81"},
    "ds_all":           {"en": "All three", "zh": "\u5168\u90e8\u4e09\u4e2a"},
    "ds_all_d":         {"en": "Download all datasets at once",
                         "zh": "\u4e00\u6b21\u6027\u4e0b\u8f7d\u5168\u90e8\u6570\u636e\u96c6"},
    "pick_ds":          {"en": "Pick a dataset", "zh": "\u9009\u62e9\u6570\u636e\u96c6"},

    # ── strategies ────────────────────────────────────────────────────────
    "strat_temporal":    {"en": "Grouped Temporal", "zh": "\u65f6\u5e8f\u5206\u7ec4"},
    "strat_temporal_d":  {"en": "Sort by time, patient-disjoint (recommended)",
                          "zh": "\u6309\u65f6\u95f4\u6392\u5e8f\uff0c\u60a3\u8005\u4e0d\u76f8\u4ea4\uff08\u63a8\u8350\uff09"},
    "strat_random":      {"en": "Grouped Random", "zh": "\u968f\u673a\u5206\u7ec4"},
    "strat_random_d":    {"en": "Random patient-disjoint split",
                          "zh": "\u968f\u673a\u60a3\u8005\u4e0d\u76f8\u4ea4\u5206\u5272"},
    "strat_stratified":  {"en": "Stratified Grouped", "zh": "\u5206\u5c42\u5206\u7ec4"},
    "strat_stratified_d":{"en": "Preserve positive rate across splits",
                          "zh": "\u4fdd\u6301\u5404\u5206\u5272\u7684\u9633\u6027\u7387\u4e00\u81f4"},
    "pick_strat":        {"en": "Split strategy", "zh": "\u5206\u5272\u7b56\u7565"},

    # ── quick start ───────────────────────────────────────────────────────
    "qs_title":         {"en": "QUICK START", "zh": "\u5feb\u901f\u5f00\u59cb"},
    "qs_desc":          {"en": "Download a real UCI dataset, split it, ready to train.",
                         "zh": "\u4e0b\u8f7d\u771f\u5b9e UCI \u6570\u636e\u96c6\uff0c\u5206\u5272\u5b8c\u6210\uff0c\u5373\u53ef\u8bad\u7ec3\u3002"},
    "qs_step1":         {"en": "Download {ds} from UCI", "zh": "\u4ece UCI \u4e0b\u8f7d {ds}"},
    "qs_step2":         {"en": "Split into train / valid / test",
                         "zh": "\u5206\u5272\u4e3a train / valid / test"},
    "qs_step3":         {"en": "Verify patient-disjoint + temporal order",
                         "zh": "\u9a8c\u8bc1\u60a3\u8005\u4e0d\u76f8\u4ea4 + \u65f6\u5e8f\u987a\u5e8f"},
    "qs_done":          {"en": "Done! Dataset split successfully.",
                         "zh": "\u5b8c\u6210\uff01\u6570\u636e\u96c6\u5206\u5272\u6210\u529f\u3002"},
    "qs_next":          {"en": "Next: select 'Full Pipeline' to train a model.",
                         "zh": "\u4e0b\u4e00\u6b65\uff1a\u9009\u62e9\u300c\u5b8c\u6574\u7ba1\u7ebf\u300d\u6765\u8bad\u7ec3\u6a21\u578b\u3002"},
    "qs_results":       {"en": "Results", "zh": "\u7ed3\u679c"},
    "download_fail":    {"en": "Download failed.", "zh": "\u4e0b\u8f7d\u5931\u8d25\u3002"},
    "split_fail":       {"en": "Split failed.", "zh": "\u5206\u5272\u5931\u8d25\u3002"},
    "downloading":      {"en": "Downloading {ds}...", "zh": "\u6b63\u5728\u4e0b\u8f7d {ds}..."},
    "splitting":        {"en": "Splitting with safety checks...",
                         "zh": "\u6b63\u5728\u5b89\u5168\u5206\u5272..."},

    # ── download ──────────────────────────────────────────────────────────
    "dl_title":         {"en": "DOWNLOAD DATASET", "zh": "\u4e0b\u8f7d\u6570\u636e\u96c6"},
    "dl_desc":          {"en": "Download real UCI medical datasets.",
                         "zh": "\u4e0b\u8f7d\u771f\u5b9e UCI \u533b\u5b66\u6570\u636e\u96c6\u3002"},
    "dl_done":          {"en": "Download Complete", "zh": "\u4e0b\u8f7d\u5b8c\u6210"},
    "which_ds":         {"en": "Which dataset?", "zh": "\u9009\u62e9\u54ea\u4e2a\u6570\u636e\u96c6\uff1f"},

    # ── split ─────────────────────────────────────────────────────────────
    "sp_title":         {"en": "SPLIT YOUR CSV", "zh": "\u5206\u5272\u4f60\u7684 CSV"},
    "sp_desc1":         {"en": "Split a CSV into train/valid/test sets.",
                         "zh": "\u5c06 CSV \u5206\u5272\u4e3a train/valid/test \u96c6\u3002"},
    "sp_desc2":         {"en": "Patient-disjoint, with medical safety guarantees.",
                         "zh": "\u60a3\u8005\u4e0d\u76f8\u4ea4\uff0c\u533b\u5b66\u5b89\u5168\u4fdd\u8bc1\u3002"},
    "sp_pick_csv":      {"en": "Select your CSV file", "zh": "\u9009\u62e9\u4f60\u7684 CSV \u6587\u4ef6"},
    "sp_manual":        {"en": "Enter path manually...", "zh": "\u624b\u52a8\u8f93\u5165\u8def\u5f84..."},
    "sp_csv_path":      {"en": "CSV path", "zh": "CSV \u8def\u5f84"},
    "sp_not_found":     {"en": "File not found.", "zh": "\u6587\u4ef6\u672a\u627e\u5230\u3002"},
    "sp_bad_csv":       {"en": "Cannot read CSV header.", "zh": "\u65e0\u6cd5\u8bfb\u53d6 CSV \u8868\u5934\u3002"},
    "sp_pick_pid":      {"en": "Which column is the Patient ID?",
                         "zh": "\u54ea\u4e00\u5217\u662f\u60a3\u8005 ID\uff1f"},
    "sp_pick_target":   {"en": "Which column is the Target (0/1)?",
                         "zh": "\u54ea\u4e00\u5217\u662f\u76ee\u6807\u53d8\u91cf (0/1)\uff1f"},
    "sp_pick_time":     {"en": "Which column is the Time/Date?",
                         "zh": "\u54ea\u4e00\u5217\u662f\u65f6\u95f4\u5217\uff1f"},
    "sp_config":        {"en": "Configuration", "zh": "\u914d\u7f6e\u786e\u8ba4"},
    "sp_run":           {"en": "Run split", "zh": "\u6267\u884c\u5206\u5272"},
    "sp_cancel":        {"en": "Cancel", "zh": "\u53d6\u6d88"},
    "sp_done":          {"en": "Split Complete", "zh": "\u5206\u5272\u5b8c\u6210"},
    "sp_saved":         {"en": "Saved to:", "zh": "\u4fdd\u5b58\u81f3\uff1a"},

    # ── pipeline ──────────────────────────────────────────────────────────
    "pl_title":         {"en": "FULL PIPELINE", "zh": "\u5b8c\u6574\u7ba1\u7ebf"},
    "pl_desc1":         {"en": "End-to-end training with 28 safety gates.",
                         "zh": "28 \u4e2a\u5b89\u5168\u95e8\u7aef\u5230\u7aef\u8bad\u7ec3\u3002"},
    "pl_desc2":         {"en": "Generates publication-grade evidence.",
                         "zh": "\u751f\u6210\u53d1\u8868\u7ea7\u8bc1\u636e\u3002"},
    "pl_mode":          {"en": "Mode", "zh": "\u6a21\u5f0f"},
    "pl_demo":          {"en": "Demo Mode", "zh": "\u6f14\u793a\u6a21\u5f0f"},
    "pl_demo_d":        {"en": "Use synthetic data -- great for first run",
                         "zh": "\u4f7f\u7528\u5408\u6210\u6570\u636e -- \u9996\u6b21\u4f7f\u7528\u63a8\u8350"},
    "pl_user":          {"en": "Your CSV", "zh": "\u4f60\u7684 CSV"},
    "pl_user_d":        {"en": "Bring your own dataset",
                         "zh": "\u4f7f\u7528\u81ea\u5df1\u7684\u6570\u636e\u96c6"},
    "pl_demo_box":      {"en": "Demo Pipeline", "zh": "\u6f14\u793a\u7ba1\u7ebf"},
    "pl_time_hint":     {"en": "This will take 3-8 minutes.",
                         "zh": "\u8fd9\u5c06\u9700\u8981 3-8 \u5206\u949f\u3002"},
    "pl_start":         {"en": "Start pipeline", "zh": "\u5f00\u59cb\u8fd0\u884c"},
    "pl_running":       {"en": "Running full pipeline...",
                         "zh": "\u6b63\u5728\u8fd0\u884c\u5b8c\u6574\u7ba1\u7ebf..."},
    "pl_done":          {"en": "Pipeline Complete", "zh": "\u7ba1\u7ebf\u5b8c\u6210"},
    "pl_fail":          {"en": "Pipeline had failures.", "zh": "\u7ba1\u7ebf\u8fd0\u884c\u5931\u8d25\u3002"},
    "pl_config":        {"en": "Pipeline Configuration", "zh": "\u7ba1\u7ebf\u914d\u7f6e"},

    # ── health ────────────────────────────────────────────────────────────
    "hc_title":         {"en": "HEALTH CHECK", "zh": "\u73af\u5883\u68c0\u67e5"},
    "hc_desc":          {"en": "Verifying your environment...",
                         "zh": "\u6b63\u5728\u68c0\u67e5\u73af\u5883..."},
    "hc_checking":      {"en": "Checking CLI...", "zh": "\u68c0\u67e5 CLI..."},
    "hc_doctor":        {"en": "Run full doctor", "zh": "\u8fd0\u884c\u5b8c\u6574\u68c0\u67e5"},
    "hc_back":          {"en": "Back", "zh": "\u8fd4\u56de"},

    # ── guide ─────────────────────────────────────────────────────────────
    "gu_title":         {"en": "GUIDE", "zh": "\u4f7f\u7528\u6307\u5357"},
    "gu_next":          {"en": "Next page", "zh": "\u4e0b\u4e00\u9875"},
    "gu_prev":          {"en": "Previous page", "zh": "\u4e0a\u4e00\u9875"},
    "gu_back":          {"en": "Back to menu", "zh": "\u8fd4\u56de\u83dc\u5355"},

    "gu_t1":            {"en": "What is Data Leakage?", "zh": "\u4ec0\u4e48\u662f\u6570\u636e\u6cc4\u6f0f\uff1f"},
    "gu_b1":            {"en": "Data leakage in medical ML means information from\noutside the intended training scope accidentally\ninfluences model training. This inflates performance\nand can lead to unsafe clinical decisions.",
                         "zh": "\u533b\u5b66 ML \u4e2d\u7684\u6570\u636e\u6cc4\u6f0f\u662f\u6307\u8bad\u7ec3\u8303\u56f4\u4e4b\u5916\u7684\u4fe1\u606f\n\u610f\u5916\u5730\u5f71\u54cd\u4e86\u6a21\u578b\u8bad\u7ec3\u3002\u8fd9\u4f1a\u865a\u9ad8\u6027\u80fd\u6307\u6807\uff0c\n\u5e76\u53ef\u80fd\u5bfc\u81f4\u4e0d\u5b89\u5168\u7684\u4e34\u5e8a\u51b3\u7b56\u3002"},
    "gu_t2":            {"en": "What This Pipeline Does", "zh": "\u8fd9\u4e2a\u7ba1\u7ebf\u505a\u4ec0\u4e48"},
    "gu_b2":            {"en": "- 28 sequential fail-closed safety gates\n- Patient-disjoint temporal splitting\n- Feature leakage detection\n- Tuning and calibration leakage guards\n- Publication-grade evidence artifacts",
                         "zh": "- 28 \u4e2a\u987a\u5e8f\u5931\u8d25\u5173\u95ed\u5b89\u5168\u95e8\n- \u60a3\u8005\u4e0d\u76f8\u4ea4\u65f6\u5e8f\u5206\u5272\n- \u7279\u5f81\u6cc4\u6f0f\u68c0\u6d4b\n- \u8c03\u53c2\u548c\u6821\u51c6\u6cc4\u6f0f\u9632\u62a4\n- \u53d1\u8868\u7ea7\u8bc1\u636e\u751f\u6210"},
    "gu_t3":            {"en": "Available Datasets", "zh": "\u53ef\u7528\u6570\u636e\u96c6"},
    "gu_b3":            {"en": "Heart Disease   -- 297 rows, 13 features\nBreast Cancer   -- 569 rows, 30 features\nKidney Disease  -- 399 rows, 24 features\n\nDownload: python3 examples/download_real_data.py heart",
                         "zh": "\u5fc3\u810f\u75c5     -- 297 \u884c, 13 \u7279\u5f81\n\u4e73\u817a\u764c     -- 569 \u884c, 30 \u7279\u5f81\n\u6162\u6027\u80be\u75c5   -- 399 \u884c, 24 \u7279\u5f81\n\n\u4e0b\u8f7d: python3 examples/download_real_data.py heart"},
    "gu_t4":            {"en": "Getting Started", "zh": "\u5feb\u901f\u4e0a\u624b"},
    "gu_b4":            {"en": "git clone https://github.com/Furinaaa-Cancan/\n    medical-ml-leakage-guard.git\ncd medical-ml-leakage-guard\npip install -r requirements.txt\npython3 scripts/mlgg.py play",
                         "zh": "git clone https://github.com/Furinaaa-Cancan/\n    medical-ml-leakage-guard.git\ncd medical-ml-leakage-guard\npip install -r requirements.txt\npython3 scripts/mlgg.py play"},

    # ── generic ───────────────────────────────────────────────────────────
    "rows":             {"en": "rows", "zh": "\u884c"},
    "patients":         {"en": "patients", "zh": "\u60a3\u8005"},
    "columns":          {"en": "columns", "zh": "\u5217"},
    "required":         {"en": "required", "zh": "\u5fc5\u9700"},
    "optional":         {"en": "optional", "zh": "\u53ef\u9009"},
    "file":             {"en": "File:", "zh": "\u6587\u4ef6\uff1a"},
    "patient":          {"en": "Patient:", "zh": "\u60a3\u8005\uff1a"},
    "target":           {"en": "Target:", "zh": "\u76ee\u6807\uff1a"},
    "time":             {"en": "Time:", "zh": "\u65f6\u95f4\uff1a"},
    "strategy":         {"en": "Strategy:", "zh": "\u7b56\u7565\uff1a"},
    "output":           {"en": "Output:", "zh": "\u8f93\u51fa\uff1a"},
    "none":             {"en": "(none)", "zh": "(\u65e0)"},
    "no_time_col":      {"en": "No remaining columns for time.",
                         "zh": "\u6ca1\u6709\u53ef\u7528\u7684\u65f6\u95f4\u5217\u3002"},
}

def t(key: str, **kwargs: Any) -> str:
    """Get translated string."""
    val = _T.get(key, {}).get(LANG, _T.get(key, {}).get("en", key))
    if kwargs:
        val = val.format(**kwargs)
    return val


def detect_lang() -> str:
    """Auto-detect language from system locale."""
    try:
        loc = locale.getlocale()[0] or os.environ.get("LANG", "")
        if loc.startswith("zh"):
            return "zh"
    except Exception:
        pass
    return "en"


# ── raw key input (with mouse support) ────────────────────────────────────────
def _getch() -> str:
    """Read a keypress. Returns 'MOUSE:row' for mouse clicks."""
    try:
        import tty, termios, select as sel_mod
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Read more of the escape sequence
                buf = ""
                while True:
                    if not sel_mod.select([fd], [], [], 0.05)[0]:
                        break
                    c = sys.stdin.read(1)
                    buf += c
                    # SGR mouse: \033[<btn;col;rowM or m
                    if buf.startswith("[<") and c in ("M", "m"):
                        # Parse SGR mouse: [<btn;col;row M
                        try:
                            parts = buf[2:-1].split(";")
                            btn = int(parts[0])
                            row = int(parts[2])
                            if btn == 0 and c == "M":  # left click press
                                return f"MOUSE:{row}"
                        except (ValueError, IndexError):
                            pass
                        return ""  # ignore other mouse events
                    # Arrow keys: [A, [B
                    if len(buf) == 2 and buf[0] == "[" and buf[1] in "ABCD":
                        return {"[A": "UP", "[B": "DOWN"}.get(buf, "ESC")
                    # Guard: very long unexpected sequences
                    if len(buf) >= 20:
                        break
                return "ESC"
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


def _cursor_row() -> int:
    """Get current cursor row (1-indexed). Returns 0 on failure."""
    try:
        import tty, termios, select as sel_mod
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdout.write(GET_CURSOR)
            sys.stdout.flush()
            buf = ""
            for _ in range(25):
                if not sel_mod.select([fd], [], [], 0.15)[0]:
                    break
                c = sys.stdin.read(1)
                buf += c
                if c == "R":
                    break
            # Drain any leftover bytes to prevent leaking into _getch()
            while sel_mod.select([fd], [], [], 0.01)[0]:
                sys.stdin.read(1)
            # Response: \033[row;colR
            if "[" in buf and ";" in buf:
                row_str = buf.split("[")[1].split(";")[0]
                return int(row_str)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        pass
    return 0


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
            sys.stdout.write(f"\r  {s('C',f)} {s('W',self.label)}"); sys.stdout.flush()
            self._stop.wait(0.12)

def run_spinner(cmd: List[str], label: str, cwd: str = "") -> Tuple[int, str, str]:
    with Spinner(label):
        p = subprocess.run(cmd, cwd=cwd or str(REPO_ROOT), capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


# ── box drawing ───────────────────────────────────────────────────────────────
def box(title: str, lines: List[str], color: str = "C", width: int = 0) -> None:
    w = width or max(max((_wlen(l) for l in lines), default=0), _wlen(title)) + 4
    print(f"  {s(color, '┌' + '─' * w + '┐')}")
    if title:
        pad = w - _wlen(title) - 2
        print(f"  {s(color, '│')} {s('W', title, bold=True)}{' ' * max(pad,0)}{s(color, '│')}")
        print(f"  {s(color, '├' + '─' * w + '┤')}")
    for line in lines:
        pad = w - _wlen(line) - 2
        print(f"  {s(color, '│')} {line}{' ' * max(pad, 0)}{s(color, '│')}")
    print(f"  {s(color, '└' + '─' * w + '┘')}")


# ── select menu (arrow keys + mouse click) ───────────────────────────────────
def select(title: str, options: List[str], descs: Optional[List[str]] = None) -> int:
    """Arrow-key + mouse-click menu. Returns 0-based index, -1 on quit."""
    sel = 0
    n = len(options)
    has_desc = descs and len(descs) == n
    menu_start_row = 0  # absolute terminal row where first option is drawn

    def _draw() -> None:
        nonlocal menu_start_row
        sys.stdout.write(HIDE_CUR)
        sys.stdout.write(MOUSE_OFF)  # disable mouse during cursor query
        sys.stdout.flush()
        print()
        if title:
            print(f"  {s('C', title, bold=True)}")
        print()
        if menu_start_row == 0:  # only query cursor position on FIRST draw
            menu_start_row = _cursor_row()
        sys.stdout.write(MOUSE_ON)
        sys.stdout.flush()
        for i in range(n):
            if i == sel:
                lbl = f" {options[i]} "
                d = f"  {descs[i]}" if has_desc and descs[i] else ""
                print(f"  {s('C','>', bold=True)} {BG['b']}{FG['W']}{BOLD}{lbl}{RST}{s('C', d)}")
            else:
                d = f"  {DIM}{descs[i]}{RST}" if has_desc and descs[i] else ""
                print(f"    {DIM}{options[i]}{RST}{d}")
        print()
        print(f"  {DIM}{t('nav_hint')}{RST}")

    def _cleanup() -> None:
        sys.stdout.write(MOUSE_OFF)
        sys.stdout.write(SHOW_CUR)
        sys.stdout.flush()

    extra = (1 if title else 0)
    line_count = n + extra + 3

    _draw()
    while True:
        key = _getch()
        if key == "UP" and sel > 0:
            sel -= 1
        elif key == "DOWN" and sel < n - 1:
            sel += 1
        elif key == "ENTER":
            _cleanup()
            return sel
        elif key in ("Q", "CTRL_C", "CTRL_D", "ESC"):
            _cleanup()
            return -1
        elif key.startswith("MOUSE:"):
            # Map click row to menu item
            try:
                click_row = int(key.split(":")[1])
                idx = click_row - menu_start_row
                if 0 <= idx < n:
                    sel = idx
                    # Redraw with selection highlighted, then confirm
                    for _ in range(line_count):
                        sys.stdout.write(f"{UP_LINE}{ERASE}")
                    sys.stdout.write("\r"); sys.stdout.flush()
                    _draw()
                    # Auto-select on click
                    _cleanup()
                    return sel
            except (ValueError, IndexError):
                pass
            continue
        elif isinstance(key, str) and len(key) == 1 and key.isdigit() and 1 <= int(key) <= n:
            sel = int(key) - 1
        else:
            continue
        for _ in range(line_count):
            sys.stdout.write(f"{UP_LINE}{ERASE}")
        sys.stdout.write("\r"); sys.stdout.flush()
        _draw()


# ── step tracker ──────────────────────────────────────────────────────────────
def render_steps(steps: List[Tuple[str, str]]) -> None:
    for i, (label, st) in enumerate(steps):
        num = f"{i+1}/{len(steps)}"
        if st == "running":
            print(f"  {s('C', num, True)}  {s('C','>>>')}  {s('W', label, True)}")
        elif st == "done":
            print(f"  {s('G', num)}  {s('G','[ok]')}  {s('W', label)}")
        elif st == "fail":
            print(f"  {s('R', num)}  {s('R','[!!]')}  {s('R', label)}")
        else:
            print(f"  {DIM}{num}  [ ]  {label}{RST}")

def erase_n(n: int) -> None:
    for _ in range(n):
        sys.stdout.write(f"{UP_LINE}{ERASE}")
    sys.stdout.write("\r"); sys.stdout.flush()


# ── file helpers ──────────────────────────────────────────────────────────────
def scan_csv() -> List[Path]:
    found: List[Path] = []
    for d in [EXAMPLES_DIR, DESKTOP, Path.home()/"Downloads", Path.home()/"Documents", REPO_ROOT, DEFAULT_OUT]:
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


# ── reusable selectors ────────────────────────────────────────────────────────
def pick_dataset(title_key: str = "pick_ds") -> int:
    return select(
        t(title_key),
        [t("ds_heart"), t("ds_breast"), t("ds_kidney")],
        [t("ds_heart_d"), t("ds_breast_d"), t("ds_kidney_d")],
    )

def pick_strategy() -> int:
    return select(
        t("pick_strat"),
        [t("strat_temporal"), t("strat_random"), t("strat_stratified")],
        [t("strat_temporal_d"), t("strat_random_d"), t("strat_stratified_d")],
    )

def pick_csv_file() -> Optional[str]:
    """Let user pick a CSV from auto-scan or manual input."""
    files = scan_csv()
    if files:
        names = [f.name for f in files]
        descs = [f"{csv_rows(f)} {t('rows')}  --  {f.parent}" for f in files]
        names.append(t("sp_manual"))
        descs.append("")
        fi = select(t("sp_pick_csv"), names, descs)
        if fi < 0:
            return None
        if fi == len(files):
            sys.stdout.write(SHOW_CUR)
            return input(f"  {s('C','>')} {s('W', t('sp_csv_path'))}: ").strip()
        return str(files[fi])
    sys.stdout.write(SHOW_CUR)
    return input(f"  {s('C','>')} {s('W', t('sp_csv_path'))}: ").strip()

def pick_columns(csv_path: str) -> Optional[Dict[str, str]]:
    """Let user pick patient_id, target, strategy, time columns. All via selection."""
    columns = csv_cols(Path(csv_path))
    if not columns:
        print(f"  {s('R', t('sp_bad_csv'))}"); return None

    rows = csv_rows(Path(csv_path))
    print(f"\n  {s('W', Path(csv_path).name, bold=True)}  {DIM}{rows} {t('rows')}, {len(columns)} {t('columns')}{RST}")

    pi = select(t("sp_pick_pid"), columns)
    if pi < 0: return None
    pid = columns[pi]

    rem1 = [c for c in columns if c != pid]
    ti = select(t("sp_pick_target"), rem1)
    if ti < 0: return None
    tgt = rem1[ti]

    si = pick_strategy()
    if si < 0: return None
    strat = ["grouped_temporal", "grouped_random", "stratified_grouped"][si]

    tcol = ""
    if strat == "grouped_temporal":
        rem2 = [c for c in columns if c not in (pid, tgt)]
        if not rem2:
            print(f"  {s('R', t('no_time_col'))}")
            return None
        tci = select(t("sp_pick_time"), rem2)
        if tci < 0: return None
        tcol = rem2[tci]

    return {"pid": pid, "target": tgt, "strategy": strat, "time": tcol}


# ── pixel logo ────────────────────────────────────────────────────────────────
LOGO = f"""
{s('C','',True)}
    ██╗     ███████╗ █████╗ ██╗  ██╗ █████╗  ██████╗ ███████╗
    ██║     ██╔════╝██╔══██╗██║ ██╔╝██╔══██╗██╔════╝ ██╔════╝
    ██║     █████╗  ███████║█████╔╝ ███████║██║  ███╗█████╗
    ██║     ██╔══╝  ██╔══██║██╔═██╗ ██╔══██║██║   ██║██╔══╝
    ███████╗███████╗██║  ██║██║  ██╗██║  ██║╚██████╔╝███████╗
    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝{RST}
{s('Y','',True)}
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
    print(f"    {DIM}{t('subtitle')}{RST}")
    print(f"    {DIM}{t('tagline')}{RST}")
    lang_label = "EN/\u4e2d" if LANG == "en" else "\u4e2d/EN"
    return select(
        "",
        [t("m_quick"), t("m_download"), t("m_split"), t("m_pipeline"),
         t("m_health"), t("m_guide"), f"{t('m_lang')} [{lang_label}]", t("m_quit")],
        [t("m_quick_d"), t("m_download_d"), t("m_split_d"), t("m_pipeline_d"),
         t("m_health_d"), t("m_guide_d"), t("m_lang_d"), t("m_quit_d")],
    )


# ── Quick Start ───────────────────────────────────────────────────────────────
def action_quick_start() -> None:
    _clear()
    box(t("qs_title"), [t("qs_desc"), f"{t('output')} {DEFAULT_OUT}/"], color="G")

    ds = pick_dataset()
    if ds < 0: return
    keys = ["heart", "breast", "ckd"]
    files = ["heart_disease", "breast_cancer", "chronic_kidney_disease"]
    labels = [t("ds_heart"), t("ds_breast"), t("ds_kidney")]
    key, fname, label = keys[ds], files[ds], labels[ds]

    steps: List[Tuple[str, str]] = [
        (t("qs_step1", ds=label), "pending"),
        (t("qs_step2"), "pending"),
        (t("qs_step3"), "pending"),
    ]
    print()
    steps[0] = (steps[0][0], "running"); render_steps(steps)

    rc, _, err = run_spinner(
        [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"), key],
        t("downloading", ds=label),
    )
    erase_n(len(steps))
    if rc != 0:
        steps[0] = (steps[0][0], "fail"); render_steps(steps)
        print(f"\n  {s('R', t('download_fail'))}"); return
    steps[0] = (steps[0][0], "done")

    csv_path = EXAMPLES_DIR / f"{fname}.csv"
    out_dir = DEFAULT_OUT / fname
    steps[1] = (steps[1][0], "running"); render_steps(steps)

    rc, _, err = run_spinner(
        [sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
         "--input", str(csv_path), "--output-dir", str(out_dir / "data"),
         "--patient-id-col", "patient_id", "--target-col", "y",
         "--time-col", "event_time", "--strategy", "grouped_temporal"],
        t("splitting"),
    )
    erase_n(len(steps))
    if rc != 0:
        steps[1] = (steps[1][0], "fail"); render_steps(steps)
        print(f"\n  {s('R', t('split_fail'))}"); return
    steps[1] = (steps[1][0], "done")
    steps[2] = (steps[2][0], "done"); render_steps(steps)

    try:
        import pandas as pd
        tr = pd.read_csv(out_dir / "data" / "train.csv")
        va = pd.read_csv(out_dir / "data" / "valid.csv")
        te = pd.read_csv(out_dir / "data" / "test.csv")
        print()
        box(t("qs_results"), [
            f"train.csv   {len(tr):>4} {t('rows')}   {tr['patient_id'].nunique():>4} {t('patients')}",
            f"valid.csv   {len(va):>4} {t('rows')}   {va['patient_id'].nunique():>4} {t('patients')}",
            f"test.csv    {len(te):>4} {t('rows')}    {te['patient_id'].nunique():>3} {t('patients')}",
            "", f"{t('sp_saved')} {out_dir}/data/",
        ], color="G")
    except Exception:
        print(f"\n  {s('G', t('qs_done'), True)} {out_dir}/data/")
    print(f"\n  {DIM}{t('qs_next')}{RST}")


# ── Download ──────────────────────────────────────────────────────────────────
def action_download() -> None:
    _clear()
    box(t("dl_title"), [t("dl_desc"), f"{t('output')} {EXAMPLES_DIR}/"], color="Y")

    ch = select(
        t("which_ds"),
        [t("ds_heart"), t("ds_breast"), t("ds_kidney"), t("ds_all")],
        [t("ds_heart_d"), t("ds_breast_d"), t("ds_kidney_d"), t("ds_all_d")],
    )
    if ch < 0: return
    key = ["heart", "breast", "ckd", "all"][ch]

    rc, out, err = run_spinner(
        [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"), key],
        t("downloading", ds=key),
    )
    if rc == 0:
        info = [l.strip() for l in (out or "").split("\n") if l.strip() and ("Output" in l or "Row" in l)]
        box(t("dl_done"), info or ["OK"], color="G")
    else:
        print(f"  {s('R', t('download_fail'))}")


# ── Split CSV ─────────────────────────────────────────────────────────────────
def action_split() -> None:
    _clear()
    box(t("sp_title"), [t("sp_desc1"), t("sp_desc2")], color="M")

    csv_path = pick_csv_file()
    if not csv_path or not Path(csv_path).exists():
        print(f"  {s('R', t('sp_not_found'))}"); return

    result = pick_columns(csv_path)
    if not result: return

    out_dir = DEFAULT_OUT / Path(csv_path).stem
    print()
    box(t("sp_config"), [
        f"{t('file')}     {Path(csv_path).name}  ({csv_rows(Path(csv_path))} {t('rows')})",
        f"{t('patient')}  {result['pid']}",
        f"{t('target')}   {result['target']}",
        f"{t('time')}     {result['time'] or t('none')}",
        f"{t('strategy')} {result['strategy']}",
        f"{t('output')}   {out_dir}/data/",
    ], color="B")

    ci = select("", [t("sp_run"), t("sp_cancel")])
    if ci != 0: return

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
        "--input", csv_path, "--output-dir", str(out_dir / "data"),
        "--patient-id-col", result["pid"], "--target-col", result["target"],
        "--strategy", result["strategy"],
    ]
    if result["time"]:
        cmd.extend(["--time-col", result["time"]])

    rc, _, err = run_spinner(cmd, t("splitting"))
    if rc == 0:
        try:
            import pandas as pd
            tr = pd.read_csv(out_dir / "data" / "train.csv")
            va = pd.read_csv(out_dir / "data" / "valid.csv")
            te = pd.read_csv(out_dir / "data" / "test.csv")
            box(t("sp_done"), [
                f"train.csv   {len(tr):>4} {t('rows')}",
                f"valid.csv   {len(va):>4} {t('rows')}",
                f"test.csv    {len(te):>4} {t('rows')}",
                "", f"{t('sp_saved')} {out_dir}/data/",
            ], color="G")
        except Exception:
            print(f"  {s('G','[ok]',True)} {out_dir}/data/")
    else:
        print(f"  {s('R', t('split_fail'))}")
        if err:
            for l in err.strip().split("\n")[-5:]: print(f"  {DIM}{l}{RST}")


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def action_full_pipeline() -> None:
    _clear()
    box(t("pl_title"), [t("pl_desc1"), t("pl_desc2")], color="Y")

    mi = select(t("pl_mode"),
                [t("pl_demo"), t("pl_user")],
                [t("pl_demo_d"), t("pl_user_d")])
    if mi < 0: return

    if mi == 0:
        project_root = str(DEFAULT_OUT / "pipeline")
        print()
        box(t("pl_demo_box"), [f"{t('output')} {project_root}/", t("pl_time_hint")], color="C")
        ci = select("", [t("pl_start"), t("sp_cancel")])
        if ci != 0: return
        rc, _, err = run_spinner(
            [sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
             "--project-root", project_root, "--mode", "guided", "--yes"],
            t("pl_running"),
        )
    else:
        csv_path = pick_csv_file()
        if not csv_path or not Path(csv_path).exists():
            print(f"  {s('R', t('sp_not_found'))}"); return
        result = pick_columns(csv_path)
        if not result: return

        project_root = str(DEFAULT_OUT / Path(csv_path).stem / "pipeline")
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "onboarding",
            "--project-root", project_root, "--mode", "guided", "--yes",
            "--input-csv", csv_path,
            "--patient-id-col", result["pid"], "--target-col", result["target"],
            "--split-strategy", result["strategy"],
        ]
        if result["time"]:
            cmd.extend(["--time-col", result["time"]])

        print()
        box(t("pl_config"), [
            f"{t('file')}     {Path(csv_path).name}",
            f"{t('patient')}  {result['pid']}  |  {t('target')} {result['target']}",
            f"{t('strategy')} {result['strategy']}",
            f"{t('output')}   {project_root}/",
        ], color="C")
        ci = select("", [t("pl_start"), t("sp_cancel")])
        if ci != 0: return
        rc, _, err = run_spinner(cmd, t("pl_running"))

    if rc == 0:
        box(t("pl_done"), [
            f"{t('output')} {project_root}/",
            "  evidence/  -- audit artifacts",
            "  models/    -- trained model",
            "  data/      -- split datasets",
        ], color="G")
    else:
        print(f"  {s('R', t('pl_fail'))}")
        if err:
            for l in (err or "").strip().split("\n")[-5:]: print(f"  {DIM}{l}{RST}")


# ── Health Check ──────────────────────────────────────────────────────────────
def action_health_check() -> None:
    _clear()
    box(t("hc_title"), [t("hc_desc")], color="G")
    print()

    checks: List[Tuple[str, bool, str]] = []
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append((f"Python {py}", sys.version_info >= (3, 9), ""))

    for pkg in ["numpy", "pandas", "sklearn", "joblib"]:
        try:
            mod = __import__(pkg)
            checks.append((f"{pkg} {getattr(mod,'__version__','?')}", True, ""))
        except ImportError:
            checks.append((pkg, False, t("required")))

    for pkg, label in [("xgboost","XGBoost"),("catboost","CatBoost"),("lightgbm","LightGBM")]:
        try:
            mod = __import__(pkg)
            checks.append((f"{label} {getattr(mod,'__version__','?')}", True, t("optional")))
        except ImportError:
            checks.append((label, False, t("optional")))

    rc, _, _ = run_spinner(
        [sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "--help"], t("hc_checking"))
    checks.append(("mlgg.py CLI", rc == 0, ""))

    for name, ok, note in checks:
        icon = s('G','[ok]') if ok else s('R','[--]')
        extra = f"  {DIM}{note}{RST}" if note else ""
        print(f"  {icon}  {name}{extra}")

    passed = sum(ok for _, ok, _ in checks)
    total = len(checks)
    w = 25; filled = int(w * passed / total)
    print(f"\n  {s('G','#'*filled)}{DIM}{'.'*(w-filled)}{RST}  {passed}/{total}\n")

    ci = select("", [t("hc_doctor"), t("hc_back")])
    if ci == 0:
        subprocess.run([sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "doctor"],
                       cwd=str(REPO_ROOT), text=True)


# ── Guide ─────────────────────────────────────────────────────────────────────
def action_guide() -> None:
    pages = [
        (t("gu_t1"), t("gu_b1")),
        (t("gu_t2"), t("gu_b2")),
        (t("gu_t3"), t("gu_b3")),
        (t("gu_t4"), t("gu_b4")),
    ]
    page = 0
    while 0 <= page < len(pages):
        _clear()
        title, body = pages[page]
        box(f"{t('gu_title')} ({page+1}/{len(pages)})", [], color="B")
        print(f"\n  {s('Y', title, bold=True)}\n")
        for l in body.split("\n"):
            print(f"    {l}")
        print()
        nav = ([t("gu_next")] if page < len(pages)-1 else []) + \
              ([t("gu_prev")] if page > 0 else []) + [t("gu_back")]
        ni = select("", nav)
        if ni < 0 or nav[ni] == t("gu_back"): break
        elif nav[ni] == t("gu_next"): page += 1
        elif nav[ni] == t("gu_prev"): page -= 1


# ── Language switch ───────────────────────────────────────────────────────────
def action_switch_lang() -> None:
    global LANG
    _clear()
    ci = select(
        t("lang_title"),
        [t("lang_en"), t("lang_zh")],
    )
    if ci == 0:
        LANG = "en"
    elif ci == 1:
        LANG = "zh"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    global LANG
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__); return 0

    # Auto-detect language, then let user confirm/change at first launch
    LANG = detect_lang()

    while True:
        try:
            ch = screen_home()
        except KeyboardInterrupt:
            print(f"\n\n  {s('C', t('bye'))}"); return 0

        if ch < 0 or ch == 7:
            print(f"\n  {s('C', t('bye'))}\n"); return 0

        actions = [action_quick_start, action_download, action_split,
                   action_full_pipeline, action_health_check, action_guide,
                   action_switch_lang]
        try:
            actions[ch]()
        except KeyboardInterrupt:
            print(f"\n  {DIM}{t('interrupted')}{RST}"); continue

        if ch == 6:  # language switch, go back to menu immediately
            continue

        print()
        sys.stdout.write(SHOW_CUR)
        try:
            input(f"  {DIM}{t('press_enter')}{RST}")
        except (EOFError, KeyboardInterrupt):
            return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        sys.stdout.write(MOUSE_OFF + SHOW_CUR); sys.stdout.flush()