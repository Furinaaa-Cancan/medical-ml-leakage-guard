#!/usr/bin/env python3
"""
ML Leakage Guard -- Interactive Pipeline Wizard.

Usage:
    python3 scripts/mlgg_pixel.py
    python3 scripts/mlgg.py play
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import locale
import os
import platform
import shutil
import subprocess
import sys
import threading

try:
    import readline  # noqa: F401 -- enables arrow-key editing in input()
except ImportError:
    pass
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
    "D": "\033[2m",
}
BG = {"b": "\033[44m", "c": "\033[46m", "g": "\033[42m", "y": "\033[43m", "k": "\033[40m"}
HIDE_CUR = "\033[?25l"
SHOW_CUR = "\033[?25h"
ERASE = "\033[2K"
UP_LINE = "\033[A"

# ── sentinel ──────────────────────────────────────────────────────────────────
BACK = type("BACK", (), {"__repr__": lambda self: "BACK"})()
SKIP = type("SKIP", (), {"__repr__": lambda self: "SKIP"})()
FAIL = type("FAIL", (), {"__repr__": lambda self: "FAIL"})()
TOTAL_STEPS = 11

def s(fg: str, text: str, bold: bool = False) -> str:
    return f"{BOLD if bold else ''}{FG.get(fg, '')}{text}{RST}"

def _wlen(text: str) -> int:
    import unicodedata, re
    clean = re.sub(r'\033\[[0-9;]*m', '', text)
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in clean)

def _cols() -> int:
    return shutil.get_terminal_size((80, 24)).columns

_TEST_MODE = bool(os.environ.get("MLGG_TEST"))
MAX_TRIALS_INPUT = 1000
MAX_OPTUNA_TRIALS_INPUT = 2000
MAX_NJOBS_INPUT = 256

def _clear() -> None:
    if _TEST_MODE:
        return
    sys.stdout.write("\033[2J\033[H"); sys.stdout.flush()

def _trunc(text: str, maxw: int) -> str:
    if _wlen(text) <= maxw:
        return text
    import unicodedata, re
    # Walk visible characters, skipping ANSI sequences
    w = 0
    i = 0
    n = len(text)
    while i < n:
        # Skip ANSI escape sequence
        if text[i] == '\033' and i + 1 < n and text[i + 1] == '[':
            j = i + 2
            while j < n and text[j] not in 'mGHJK':
                j += 1
            i = j + 1
            continue
        ch = text[i]
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw + 3 > maxw:
            return text[:i] + RST + "..."
        w += cw
        i += 1
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  i18n
# ══════════════════════════════════════════════════════════════════════════════

LANG = "en"

_T: Dict[str, Dict[str, str]] = {
    "lang_title":    {"en": "Language / \u8bed\u8a00", "zh": "Language / \u8bed\u8a00"},
    "lang_en":       {"en": "English", "zh": "English"},
    "lang_zh":       {"en": "\u4e2d\u6587", "zh": "\u4e2d\u6587"},

    "nav":           {"en": "[\u2191\u2193] move  [Enter] next  [\u2190/q] back",
                      "zh": "[\u2191\u2193] \u79fb\u52a8  [Enter] \u4e0b\u4e00\u6b65  [\u2190/q] \u8fd4\u56de"},
    "nav_first":     {"en": "[\u2191\u2193] move  [Enter] next  [\u2190/q] quit",
                      "zh": "[\u2191\u2193] \u79fb\u52a8  [Enter] \u4e0b\u4e00\u6b65  [\u2190/q] \u9000\u51fa"},
    "ms_hint":       {"en": "[\u2191\u2193] move  [Space] check/uncheck  [Enter] confirm  [a] all  [\u2190/q] back",
                      "zh": "[\u2191\u2193] \u79fb\u52a8  [\u7a7a\u683c] \u52fe\u9009/\u53d6\u6d88\u52fe\u9009  [Enter] \u786e\u8ba4  [a] \u5168\u9009  [\u2190/q] \u8fd4\u56de"},
    "bye":           {"en": "Bye!", "zh": "\u518d\u89c1\uff01"},
    "interrupted":   {"en": "Interrupted.", "zh": "\u5df2\u4e2d\u65ad\u3002"},
    "enter_continue":{"en": "Press Enter to continue...",
                      "zh": "\u6309 Enter \u7ee7\u7eed..."},
    "msg_positive_int":{"en": "Please enter a positive integer.",
                        "zh": "\u8bf7\u8f93\u5165\u6b63\u6574\u6570\u3002"},
    "msg_njobs_int": {"en": "Please enter -1 or an integer >= 1.",
                      "zh": "\u8bf7\u8f93\u5165 -1 \u6216 >=1 \u7684\u6574\u6570\u3002"},
    "msg_trials_range":{"en": f"Value out of range. Allowed: 1-{MAX_TRIALS_INPUT}.",
                        "zh": f"\u6570\u503c\u8d85\u51fa\u8303\u56f4\uff0c\u53ef\u7528\u533a\u95f4\uff1a1-{MAX_TRIALS_INPUT}\u3002"},
    "msg_optuna_trials_range":{"en": f"Value out of range. Allowed: 1-{MAX_OPTUNA_TRIALS_INPUT}.",
                               "zh": f"\u6570\u503c\u8d85\u51fa\u8303\u56f4\uff0c\u53ef\u7528\u533a\u95f4\uff1a1-{MAX_OPTUNA_TRIALS_INPUT}\u3002"},
    "msg_njobs_range":{"en": f"Value out of range. Allowed: -1 or 1-{MAX_NJOBS_INPUT}.",
                       "zh": f"\u6570\u503c\u8d85\u51fa\u8303\u56f4\uff0c\u53ef\u7528\u533a\u95f4\uff1a-1 \u6216 1-{MAX_NJOBS_INPUT}\u3002"},

    "s_lang":        {"en": "Language", "zh": "\u8bed\u8a00"},
    "s_source":      {"en": "Data Source", "zh": "\u6570\u636e\u6765\u6e90"},
    "s_dataset":     {"en": "Dataset", "zh": "\u6570\u636e\u96c6"},
    "s_config":      {"en": "Columns", "zh": "\u5217\u914d\u7f6e"},
    "s_split":       {"en": "Split", "zh": "\u5206\u5272\u914d\u7f6e"},
    "s_models":      {"en": "Models", "zh": "\u6a21\u578b\u9009\u62e9"},
    "s_tuning":      {"en": "Tuning", "zh": "\u8c03\u4f18\u914d\u7f6e"},
    "s_advanced":   {"en": "Advanced", "zh": "\u9ad8\u7ea7\u8bbe\u7f6e"},
    "s_confirm":     {"en": "Confirm", "zh": "\u786e\u8ba4"},
    "s_run":         {"en": "Execute", "zh": "\u6267\u884c"},

    "src_download":  {"en": "Built-in test datasets", "zh": "\u5185\u7f6e\u6d4b\u8bd5\u6570\u636e\u96c6"},
    "src_download_d":{"en": "Choose from curated UCI medical datasets",
                      "zh": "\u4ece\u9884\u7f6e UCI \u533b\u5b66\u6570\u636e\u96c6\u4e2d\u9009\u62e9"},
    "src_csv":       {"en": "Use your own dataset (CSV)", "zh": "\u4f7f\u7528\u81ea\u5df1\u7684\u6570\u636e\u96c6\uff08CSV\uff09"},
    "src_csv_d":     {"en": "Load your local CSV and map columns",
                      "zh": "\u52a0\u8f7d\u672c\u5730 CSV \u5e76\u914d\u7f6e\u5b57\u6bb5\u6620\u5c04"},
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
    "ds_hepatitis":  {"en": "Hepatitis", "zh": "\u809d\u708e"},
    "ds_hepatitis_d":{"en": "UCI Hepatitis -- 155 patients, 19 features",
                      "zh": "UCI \u809d\u708e -- 155 \u4f8b, 19 \u7279\u5f81"},
    "ds_spect":      {"en": "SPECT Heart", "zh": "SPECT \u5fc3\u810f"},
    "ds_spect_d":    {"en": "UCI SPECT -- 267 patients, 22 features",
                      "zh": "UCI SPECT -- 267 \u4f8b, 22 \u7279\u5f81"},
    "ds_dermatology":{"en": "Dermatology", "zh": "\u76ae\u80a4\u75c5"},
    "ds_dermatology_d":{"en": "UCI Dermatology -- 366 patients, 34 features",
                        "zh": "UCI \u76ae\u80a4\u75c5 -- 366 \u4f8b, 34 \u7279\u5f81"},

    "src_full":      {"en": "Full Publication-Grade Pipeline", "zh": "\u5b8c\u6574\u51fa\u7248\u7ea7\u7ba1\u7ebf"},
    "src_full_d":    {"en": "28-gate strict pipeline with all specs",
                      "zh": "28 \u5173 gate \u4e25\u683c\u7ba1\u7ebf\uff0c\u5305\u542b\u6240\u6709\u914d\u7f6e"},
    "src_repeat":    {"en": "Repeat last run", "zh": "\u91cd\u590d\u4e0a\u6b21\u8fd0\u884c"},
    "src_repeat_d":  {"en": "Reload previous configuration", "zh": "\u52a0\u8f7d\u4e0a\u6b21\u914d\u7f6e"},
    "hist_saved":    {"en": "Run saved to history.", "zh": "\u8fd0\u884c\u5df2\u4fdd\u5b58\u5230\u5386\u53f2\u8bb0\u5f55\u3002"},

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
    "pick_outname": {"en": "Project name (supports Chinese)",
                     "zh": "\u9879\u76ee\u540d\u79f0\uff08\u652f\u6301\u4e2d\u6587\uff09"},

    "strat_temporal":   {"en": "Grouped Temporal", "zh": "\u65f6\u5e8f\u5206\u7ec4"},
    "strat_temporal_d": {"en": "Time-ordered split for true longitudinal prediction",
                         "zh": "\u7528\u4e8e\u771f\u5b9e\u7eb5\u5411\u9884\u6d4b\u7684\u65f6\u95f4\u987a\u5e8f\u5206\u5272"},
    "strat_random":     {"en": "Grouped Random", "zh": "\u968f\u673a\u5206\u7ec4"},
    "strat_random_d":   {"en": "Random patient-disjoint split",
                         "zh": "\u968f\u673a\u60a3\u8005\u4e0d\u76f8\u4ea4\u5206\u5272"},
    "strat_stratified":   {"en": "Stratified Grouped", "zh": "\u5206\u5c42\u5206\u7ec4"},
    "strat_stratified_d": {"en": "Patient-disjoint split with stable class prevalence",
                           "zh": "\u60a3\u8005\u4e0d\u91cd\u53e0\u4e14\u4fdd\u6301\u9633\u6027\u7387\u7a33\u5b9a\u7684\u5206\u5272"},
    "split_help_title":   {"en": "How to choose", "zh": "\u600e\u4e48\u9009"},
    "split_help_temporal":{"en": "Grouped Temporal: use only when timestamps reflect real prediction order.",
                           "zh": "\u65f6\u5e8f\u5206\u7ec4\uff1a\u53ea\u5728\u65f6\u95f4\u6233\u80fd\u4ee3\u8868\u771f\u5b9e\u9884\u6d4b\u987a\u5e8f\u65f6\u4f7f\u7528\u3002"},
    "split_help_stratified":{"en": "Stratified Grouped: preferred for cross-sectional/single-visit datasets.",
                             "zh": "\u5206\u5c42\u5206\u7ec4\uff1a\u66f4\u9002\u5408\u6a2a\u65ad\u9762/\u5355\u6b21\u5c31\u8bca\u6570\u636e\u3002"},
    "split_help_default_csv":{"en": "Default first option for CSV is Stratified Grouped.",
                              "zh": "CSV \u6570\u636e\u9ed8\u8ba4\u9996\u9009\u4e3a\u5206\u5c42\u5206\u7ec4\u3002"},
    "split_help_default_download":{"en": "Built-in test datasets default to Stratified Grouped.",
                                   "zh": "\u5185\u7f6e\u6d4b\u8bd5\u6570\u636e\u96c6\u9ed8\u8ba4\u4e3a\u5206\u5c42\u5206\u7ec4\u3002"},

    "pick_ratio":    {"en": "Train / Valid / Test ratio", "zh": "\u8bad\u7ec3 / \u9a8c\u8bc1 / \u6d4b\u8bd5 \u6bd4\u4f8b"},
    "ratio_60":      {"en": "60 / 20 / 20  (standard)", "zh": "60 / 20 / 20  \uff08\u6807\u51c6\uff09"},
    "ratio_70":      {"en": "70 / 15 / 15  (more training)", "zh": "70 / 15 / 15  \uff08\u66f4\u591a\u8bad\u7ec3\uff09"},
    "ratio_70_20_10": {"en": "70 / 20 / 10  (large valid)", "zh": "70 / 20 / 10  \uff08\u5927\u9a8c\u8bc1\u96c6\uff09"},
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
    "m_lgbm":        {"en": "LightGBM (optional)", "zh": "LightGBM\uff08\u53ef\u9009\uff09"},
    "m_tabpfn":      {"en": "TabPFN (optional, ≤1k rows)", "zh": "TabPFN\uff08\u53ef\u9009\uff0c\u22641k\u884c\uff09"},

    "pick_tuning":   {"en": "Hyperparameter search", "zh": "\u8d85\u53c2\u6570\u641c\u7d22\u7b56\u7565"},
    "tune_fixed":    {"en": "Fixed Grid", "zh": "\u56fa\u5b9a\u7f51\u683c"},
    "tune_fixed_d":  {"en": "Predefined hyperparameters, fastest",
                      "zh": "\u9884\u5b9a\u4e49\u8d85\u53c2\u6570\uff0c\u6700\u5feb"},
    "tune_random":   {"en": "Random Search", "zh": "\u968f\u673a\u641c\u7d22"},
    "tune_random_d": {"en": "Sample N random combinations per family",
                      "zh": "\u6bcf\u4e2a\u6a21\u578b\u65cf\u968f\u673a\u91c7\u6837 N \u7ec4\u8d85\u53c2\u6570"},
    "tune_optuna":   {"en": "Bayesian Optimization (Optuna)", "zh": "\u8d1d\u53f6\u65af\u4f18\u5316\uff08Optuna\uff09"},
    "tune_optuna_d": {"en": "Smart sequential search, needs optuna package",
                      "zh": "\u667a\u80fd\u5e8f\u8d2f\u641c\u7d22\uff0c\u9700\u5b89\u88c5 optuna"},

    "pick_calib":    {"en": "Probability calibration", "zh": "\u6982\u7387\u6821\u51c6\u65b9\u6cd5"},
    "calib_none":    {"en": "None", "zh": "\u65e0"},
    "calib_sig":     {"en": "Sigmoid (Platt)", "zh": "Sigmoid (Platt)"},
    "calib_iso":     {"en": "Isotonic", "zh": "Isotonic\uff08\u4fdd\u5e8f\uff09"},
    "calib_power":   {"en": "Power calibration", "zh": "Power \u6821\u51c6"},
    "calib_beta":    {"en": "Beta calibration", "zh": "Beta \u6821\u51c6"},
    "calib_none_d":  {"en": "Fastest; keep raw probabilities (when model is already well-calibrated)",
                      "zh": "\u6700\u5feb\uff1b\u4fdd\u6301\u539f\u59cb\u6982\u7387\uff08\u6a21\u578b\u5df2\u8f83\u597d\u6821\u51c6\u65f6\uff09"},
    "calib_sig_d":   {"en": "Most stable on small/medium datasets (recommended default)",
                      "zh": "\u5c0f/\u4e2d\u6570\u636e\u96c6\u4e0b\u6700\u7a33\u5b9a\uff08\u9ed8\u8ba4\u63a8\u8350\uff09"},
    "calib_iso_d":   {"en": "Flexible non-linear mapping; needs more samples to avoid overfitting",
                      "zh": "\u7075\u6d3b\u7684\u975e\u7ebf\u6027\u6620\u5c04\uff1b\u9700\u8981\u66f4\u591a\u6837\u672c\u9632\u6b62\u8fc7\u62df\u5408"},
    "calib_power_d": {"en": "Conservative shape adjustment; often robust for clinical probabilities",
                      "zh": "\u4fdd\u5b88\u7684\u5f62\u72b6\u8c03\u6574\uff1b\u5bf9\u4e34\u5e8a\u6982\u7387\u901a\u5e38\u66f4\u7a33\u5b9a"},
    "calib_beta_d":  {"en": "More expressive transform; use when sample size is sufficient",
                      "zh": "\u8868\u8fbe\u80fd\u529b\u66f4\u5f3a\uff1b\u5efa\u8bae\u5728\u6837\u672c\u8db3\u591f\u65f6\u4f7f\u7528"},

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
    "c_export":      {"en": "Export CLI Command", "zh": "\u5bfc\u51fa CLI \u547d\u4ee4"},

    "adv_ask":       {"en": "Configure advanced settings?", "zh": "\u914d\u7f6e\u9ad8\u7ea7\u8bbe\u7f6e\uff1f"},
    "adv_yes":       {"en": "Yes, customize", "zh": "\u662f\uff0c\u81ea\u5b9a\u4e49"},
    "adv_no":        {"en": "No, use defaults", "zh": "\u5426\uff0c\u4f7f\u7528\u9ed8\u8ba4\u503c"},
    "adv_ignore":    {"en": "Ignore columns (comma-separated, non-feature columns to exclude):",
                      "zh": "\u5ffd\u7565\u5217\uff08\u9017\u53f7\u5206\u9694\uff0c\u6392\u9664\u7684\u975e\u7279\u5f81\u5217\uff09\uff1a"},
    "adv_ignore_mode_title": {"en": "How to edit ignore columns", "zh": "\u5ffd\u7565\u5217\u7f16\u8f91\u65b9\u5f0f"},
    "adv_ignore_mode_select": {"en": "Select from detected columns (recommended)", "zh": "\u4ece\u68c0\u6d4b\u5230\u7684\u5217\u4e2d\u9009\u62e9\uff08\u63a8\u8350\uff09"},
    "adv_ignore_mode_manual": {"en": "Manual comma-separated input", "zh": "\u624b\u52a8\u8f93\u5165\uff08\u9017\u53f7\u5206\u9694\uff09"},
    "adv_ignore_pick_columns": {"en": "Select columns to ignore", "zh": "\u9009\u62e9\u8981\u5ffd\u7565\u7684\u5217"},
    "adv_ignore_no_columns": {"en": "No detected columns available; switch to manual input.",
                              "zh": "\u672a\u68c0\u6d4b\u5230\u53ef\u9009\u5217\uff0c\u8bf7\u6539\u7528\u624b\u52a8\u8f93\u5165\u3002"},
    "adv_ignore_default_applied": {"en": "No columns detected yet. Applied safe defaults (patient/time).",
                                   "zh": "\u5c1a\u672a\u68c0\u6d4b\u5230\u5217\uff0c\u5df2\u81ea\u52a8\u5e94\u7528\u5b89\u5168\u9ed8\u8ba4\uff08patient/time\uff09\u3002"},
    "adv_ignore_manual_hint": {"en": "Tip: press Enter to keep current value; enter q to go back.",
                               "zh": "\u63d0\u793a\uff1a\u76f4\u63a5\u56de\u8f66\u4fdd\u7559\u5f53\u524d\u503c\uff1b\u8f93\u5165 q \u8fd4\u56de\u4e0a\u4e00\u6b65\u3002"},
    "adv_njobs":     {"en": "CPU workers (-1 = all cores):", "zh": "CPU \u5de5\u4f5c\u8fdb\u7a0b\uff08-1 = \u6240\u6709\u6838\u5fc3\uff09\uff1a"},
    "adv_trials":    {"en": "Max tries per model (higher = slower):", "zh": "\u6bcf\u4e2a\u6a21\u578b\u6700\u591a\u5c1d\u8bd5\u6b21\u6570\uff08\u8d8a\u5927\u8d8a\u6162\uff09\uff1a"},
    "pick_trials_preset": {"en": "Pick max tries per model", "zh": "\u9009\u62e9\u6bcf\u4e2a\u6a21\u578b\u7684\u6700\u591a\u5c1d\u8bd5\u6b21\u6570"},
    "trials_custom": {"en": "Custom value...", "zh": "\u81ea\u5b9a\u4e49\u6570\u503c..."},
    "adv_optional":  {"en": "Optional model backend policy", "zh": "\u53ef\u9009\u6a21\u578b\u540e\u7aef\u7b56\u7565"},
    "adv_optional_enable": {"en": "Keep optional models in current model pool",
                            "zh": "\u4fdd\u7559\u5f53\u524d\u6a21\u578b\u6c60\u4e2d\u7684\u53ef\u9009\u6a21\u578b"},
    "adv_optional_enable_d": {"en": "Missing dependencies will be handled at runtime (install or downgrade).",
                              "zh": "\u82e5\u7f3a\u5c11\u4f9d\u8d56\uff0c\u8fd0\u884c\u524d\u4f1a\u63d0\u793a\u5b89\u88c5\u6216\u964d\u7ea7\u3002"},
    "adv_optional_disable": {"en": "Disable optional models in current model pool",
                             "zh": "\u7981\u7528\u5f53\u524d\u6a21\u578b\u6c60\u4e2d\u7684\u53ef\u9009\u6a21\u578b"},
    "adv_optional_disable_d": {"en": "Remove xgboost/catboost/lightgbm/tabpfn from this run.",
                               "zh": "\u4ece\u672c\u6b21\u8fd0\u884c\u4e2d\u79fb\u9664 xgboost/catboost/lightgbm/tabpfn\u3002"},
    "adv_optional_removed_notice": {"en": "Removed optional models:",
                                    "zh": "\u5df2\u79fb\u9664\u53ef\u9009\u6a21\u578b\uff1a"},
    "adv_menu_title":{"en": "Advanced settings (editable)", "zh": "\u9ad8\u7ea7\u8bbe\u7f6e\uff08\u53ef\u7f16\u8f91\uff09"},
    "adv_edit_ignore":{"en": "Edit ignore columns", "zh": "\u7f16\u8f91\u5ffd\u7565\u5217"},
    "adv_edit_njobs":{"en": "Set CPU workers", "zh": "\u8bbe\u7f6e CPU \u5e76\u884c\u6570"},
    "adv_edit_optional":{"en": "Optional model backends", "zh": "\u53ef\u9009\u6a21\u578b\u540e\u7aef"},
    "adv_done":      {"en": "Done and continue", "zh": "\u5b8c\u6210\u5e76\u7ee7\u7eed"},
    "adv_current":   {"en": "Current values", "zh": "\u5f53\u524d\u914d\u7f6e"},
    "adv_njobs_auto":{"en": "Auto (-1, all cores)", "zh": "\u81ea\u52a8\uff08-1\uff0c\u5168\u6838\u5fc3\uff09"},
    "adv_njobs_1":   {"en": "1 worker (most stable)", "zh": "1 \u4e2a\u8fdb\u7a0b\uff08\u6700\u7a33\u5b9a\uff09"},
    "adv_njobs_4":   {"en": "4 workers", "zh": "4 \u4e2a\u8fdb\u7a0b"},
    "adv_njobs_8":   {"en": "8 workers", "zh": "8 \u4e2a\u8fdb\u7a0b"},
    "adv_njobs_custom":{"en": "Custom value...", "zh": "\u81ea\u5b9a\u4e49\u6570\u503c..."},
    "dep_fix_title": {"en": "Runtime dependency check", "zh": "\u8fd0\u884c\u65f6\u4f9d\u8d56\u68c0\u67e5"},
    "dep_missing_optional": {"en": "Missing optional model backends:", "zh": "\u7f3a\u5c11\u7684\u53ef\u9009\u6a21\u578b\u540e\u7aef\uff1a"},
    "dep_missing_optuna": {"en": "Optuna is required by current tuning mode but not installed.", "zh": "\u5f53\u524d\u8c03\u4f18\u6a21\u5f0f\u9700\u8981 Optuna\uff0c\u4f46\u672a\u5b89\u88c5\u3002"},
    "dep_action_install": {"en": "Auto-install missing dependencies (one-click)", "zh": "\u81ea\u52a8\u5b89\u88c5\u7f3a\u5931\u4f9d\u8d56\uff08\u4e00\u952e\uff09"},
    "dep_action_retry_failed": {"en": "Retry failed packages only", "zh": "\u4ec5\u91cd\u8bd5\u5931\u8d25\u7684\u5305"},
    "dep_action_downgrade": {"en": "Auto-downgrade and continue training", "zh": "\u81ea\u52a8\u964d\u7ea7\u5e76\u7ee7\u7eed\u8bad\u7ec3"},
    "dep_action_cancel": {"en": "Cancel run", "zh": "\u53d6\u6d88\u8fd0\u884c"},
    "dep_installing": {"en": "Installing missing dependencies...", "zh": "\u6b63\u5728\u5b89\u88c5\u7f3a\u5931\u4f9d\u8d56..."},
    "dep_installing_pkg": {"en": "Installing package: {pkg}", "zh": "\u6b63\u5728\u5b89\u88c5\u5305\uff1a{pkg}"},
    "dep_install_success": {"en": "Dependency install completed.", "zh": "\u4f9d\u8d56\u5b89\u88c5\u5b8c\u6210\u3002"},
    "dep_install_failed": {"en": "Dependency install failed.", "zh": "\u4f9d\u8d56\u5b89\u88c5\u5931\u8d25\u3002"},
    "dep_install_partial": {"en": "Some packages failed to install.", "zh": "\u90e8\u5206\u5305\u5b89\u88c5\u5931\u8d25\u3002"},
    "dep_install_ok_pkgs": {"en": "Installed successfully:", "zh": "\u5b89\u88c5\u6210\u529f\uff1a"},
    "dep_install_failed_pkgs": {"en": "Failed packages:", "zh": "\u5b89\u88c5\u5931\u8d25\u7684\u5305\uff1a"},
    "dep_downgrade_optional_removed": {"en": "Removed unavailable optional models:", "zh": "\u5df2\u79fb\u9664\u4e0d\u53ef\u7528\u53ef\u9009\u6a21\u578b\uff1a"},
    "dep_downgrade_optuna": {"en": "Downgraded tuning mode: optuna -> random_subsample.", "zh": "\u5df2\u964d\u7ea7\u8c03\u4f18\u6a21\u5f0f\uff1aoptuna -> random_subsample\u3002"},
    "dep_downgrade_model_pool": {"en": "Continuing with model pool:", "zh": "\u7ee7\u7eed\u4f7f\u7528\u6a21\u578b\u6c60\uff1a"},
    "dep_downgrade_fallback": {"en": "No usable model remained; auto-fallback to logistic_l2.", "zh": "\u65e0\u53ef\u7528\u6a21\u578b\uff0c\u5df2\u81ea\u52a8\u56de\u9000\u5230 logistic_l2\u3002"},
    "dep_cancelled": {"en": "Run cancelled due to unresolved dependencies.", "zh": "\u7531\u4e8e\u4f9d\u8d56\u672a\u89e3\u51b3\uff0c\u8fd0\u884c\u5df2\u53d6\u6d88\u3002"},
    "dep_install_cmd": {"en": "Install command:", "zh": "\u5b89\u88c5\u547d\u4ee4\uff1a"},

    "x_download":    {"en": "Downloading {ds}...", "zh": "\u6b63\u5728\u4e0b\u8f7d {ds}..."},
    "x_split":       {"en": "Splitting with safety checks...",
                      "zh": "\u6b63\u5728\u5b89\u5168\u5206\u5272..."},
    "x_train":       {"en": "Training {n} model(s)...",
                      "zh": "\u6b63\u5728\u8bad\u7ec3 {n} \u4e2a\u6a21\u578b..."},
    "x_pipeline":    {"en": "Running full pipeline...",
                      "zh": "\u6b63\u5728\u8fd0\u884c\u5b8c\u6574\u7ba1\u7ebf..."},
    "x_pipeline_full": {"en": "Running 28-gate publication pipeline...",
                        "zh": "\u6b63\u5728\u8fd0\u884c 28 \u5173\u51fa\u7248\u7ea7\u7ba1\u7ebf..."},
    "x_fail":        {"en": "Failed.", "zh": "\u5931\u8d25\u3002"},

    "r_done":        {"en": "Complete!", "zh": "\u5b8c\u6210\uff01"},
    "r_split_ok":    {"en": "Split Complete!", "zh": "\u5206\u5272\u5b8c\u6210\uff01"},
    "r_train_ok":    {"en": "Training Complete!", "zh": "\u8bad\u7ec3\u5b8c\u6210\uff01"},
    "r_metrics":     {"en": "Key Metrics (test set)", "zh": "\u5173\u952e\u6307\u6807\uff08\u6d4b\u8bd5\u96c6\uff09"},
    "r_quick_readiness": {"en": "Quick Readiness (play mode)", "zh": "\u5feb\u901f\u5c31\u7eea\u68c0\u67e5\uff08play \u6a21\u5f0f\uff09"},
    "r_play_status_not_ready": {"en": "NOT READY (play)", "zh": "\u672a\u5c31\u7eea\uff08play\uff09"},
    "r_play_status_warn": {"en": "CAUTION (play)", "zh": "\u9700\u8c28\u614e\uff08play\uff09"},
    "r_play_status_pass": {"en": "GOOD (play)", "zh": "\u826f\u597d\uff08play\uff09"},
    "r_play_blocking_fail": {"en": "Play run failed due to blocking readiness items.",
                             "zh": "play \u8fd0\u884c\u56e0\u5c31\u7eea\u963b\u65ad\u9879\u5931\u8d25\u3002"},
    "r_play_readiness_unavailable": {
        "en": "Quick readiness could not be evaluated.",
        "zh": "\u65e0\u6cd5\u5b8c\u6210 quick readiness \u8bc4\u4f30\u3002",
    },
    "r_play_readiness_reason": {"en": "Reason:", "zh": "\u539f\u56e0\uff1a"},
    "r_pub_gate_not_run_label": {"en": "Publication gate", "zh": "\u51fa\u7248\u7ea7\u95e8\u63a7"},
    "r_pub_gate_not_run_value": {"en": "NOT RUN (use workflow --strict)", "zh": "\u672a\u8fd0\u884c\uff08\u8bf7\u7528 workflow --strict\uff09"},
    "r_verdict_not_ready": {"en": "Not strict release-ready", "zh": "\u672a\u8fbe\u4e25\u683c\u53d1\u5e03\u6761\u4ef6"},
    "r_verdict_warn": {"en": "Usable with caution (play mode)", "zh": "\u53ef\u8c28\u614e\u4f7f\u7528\uff08play \u6a21\u5f0f\uff09"},
    "r_verdict_pass": {"en": "Pass in play mode (strict gate not run)", "zh": "play \u6a21\u5f0f\u901a\u8fc7\uff08\u672a\u8fd0\u884c\u4e25\u683c\u95e8\u63a7\uff09"},
    "r_saved":       {"en": "Saved to:", "zh": "\u4fdd\u5b58\u81f3\uff1a"},
    "r_next":        {"en": "All done! Results saved to output directory.",
                      "zh": "\u5168\u90e8\u5b8c\u6210\uff01\u7ed3\u679c\u5df2\u4fdd\u5b58\u81f3\u8f93\u51fa\u76ee\u5f55\u3002"},
    "r_dry_done":    {"en": "Dry-run complete. Remove --dry-run to execute.",
                      "zh": "\u5f69\u6392\u5b8c\u6210\u3002\u79fb\u9664 --dry-run \u5373\u53ef\u6267\u884c\u3002"},

    "rows":          {"en": "rows", "zh": "\u884c"},
    "patients":      {"en": "patients", "zh": "\u60a3\u8005"},
    "columns":       {"en": "columns", "zh": "\u5217"},

    "pick_valid_method": {"en": "Validation method", "zh": "\u9a8c\u8bc1\u65b9\u5f0f"},
    "valid_holdout":     {"en": "Hold-out validation set", "zh": "\u7559\u51fa\u9a8c\u8bc1\u96c6"},
    "valid_holdout_d":   {"en": "Separate fixed validation split",
                          "zh": "\u56fa\u5b9a\u7684\u72ec\u7acb\u9a8c\u8bc1\u96c6\u5206\u5272"},
    "valid_cv":          {"en": "K-fold Cross-Validation", "zh": "K \u6298\u4ea4\u53c9\u9a8c\u8bc1"},
    "valid_cv_d":        {"en": "Recommended for small datasets (<1000 rows)",
                          "zh": "\u5c0f\u6570\u636e\u96c6\u63a8\u8350\uff08<1000 \u884c\uff09"},
    "pick_cv_folds":     {"en": "Number of CV folds", "zh": "\u4ea4\u53c9\u9a8c\u8bc1\u6298\u6570"},
    "cv_3":              {"en": "3-fold  (fastest)", "zh": "3 \u6298\uff08\u6700\u5feb\uff09"},
    "cv_5":              {"en": "5-fold  (recommended)", "zh": "5 \u6298\uff08\u63a8\u8350\uff09"},
    "cv_10":             {"en": "10-fold  (most stable)", "zh": "10 \u6298\uff08\u6700\u7a33\u5b9a\uff09"},
    "pick_tt_ratio":     {"en": "Train / Test ratio (valid from CV)",
                          "zh": "\u8bad\u7ec3 / \u6d4b\u8bd5 \u6bd4\u4f8b\uff08\u9a8c\u8bc1\u7528\u4ea4\u53c9\u9a8c\u8bc1\uff09"},
    "tt_70_30":          {"en": "70 / 30  (60+10 valid / 30 test)", "zh": "70 / 30\uff0860+10\u9a8c\u8bc1 / 30\u6d4b\u8bd5\uff09"},
    "tt_80_20":          {"en": "80 / 20  (70+10 valid / 20 test)", "zh": "80 / 20\uff0870+10\u9a8c\u8bc1 / 20\u6d4b\u8bd5\uff09"},

    "s_imbalance":       {"en": "Class Imbalance", "zh": "\u7c7b\u522b\u4e0d\u5e73\u8861"},
    "pick_imbalance":    {"en": "Imbalance handling strategy",
                          "zh": "\u4e0d\u5e73\u8861\u5904\u7406\u7b56\u7565"},
    "imb_none":          {"en": "No special handling", "zh": "\u4e0d\u505a\u7279\u6b8a\u5904\u7406"},
    "imb_none_d":        {"en": "Use raw class distribution",
                          "zh": "\u4f7f\u7528\u539f\u59cb\u7c7b\u522b\u5206\u5e03"},
    "imb_auto":          {"en": "Auto (balanced when ratio \u22651.5)",
                          "zh": "\u81ea\u52a8\uff08\u6bd4\u4f8b\u22651.5 \u65f6\u5e73\u8861\uff09"},
    "imb_auto_d":        {"en": "Auto-detect imbalance and apply class_weight=balanced",
                          "zh": "\u81ea\u52a8\u68c0\u6d4b\u4e0d\u5e73\u8861\u5e76\u5e94\u7528 class_weight=balanced"},
    "imb_weight":        {"en": "Class weight balancing", "zh": "\u7c7b\u522b\u6743\u91cd\u5e73\u8861"},
    "imb_weight_d":      {"en": "Always apply class_weight=balanced to all models",
                          "zh": "\u59cb\u7ec8\u5bf9\u6240\u6709\u6a21\u578b\u5e94\u7528 class_weight=balanced"},
    "imb_smote":         {"en": "SMOTE oversampling", "zh": "SMOTE \u8fc7\u91c7\u6837"},
    "imb_smote_d":       {"en": "Synthetic minority interpolation (train fold only)",
                          "zh": "\u8bad\u7ec3\u6298\u5185\u5bf9\u5c11\u6570\u7c7b\u505a\u63d2\u503c\u8fc7\u91c7\u6837"},
    "imb_ros":           {"en": "Random oversampling", "zh": "\u968f\u673a\u8fc7\u91c7\u6837"},
    "imb_ros_d":         {"en": "Duplicate minority samples to rebalance (train fold only)",
                          "zh": "\u8bad\u7ec3\u6298\u5185\u590d\u5236\u5c11\u6570\u7c7b\u6837\u672c"},
    "imb_rus":           {"en": "Random undersampling", "zh": "\u968f\u673a\u6b20\u91c7\u6837"},
    "imb_rus_d":         {"en": "Downsample majority class (train fold only)",
                          "zh": "\u8bad\u7ec3\u6298\u5185\u4e0b\u91c7\u6837\u591a\u6570\u7c7b"},
    "imb_adasyn":        {"en": "ADASYN adaptive oversampling",
                          "zh": "ADASYN \u81ea\u9002\u5e94\u8fc7\u91c7\u6837"},
    "imb_adasyn_d":      {"en": "Adaptive synthetic oversampling by local difficulty (train fold only)",
                          "zh": "\u57fa\u4e8e\u5c40\u90e8\u96be\u5ea6\u7684\u81ea\u9002\u5e94\u5408\u6210\u8fc7\u91c7\u6837"},
    "pick_imb_metric":   {"en": "Metric for selecting best imbalance strategy", "zh": "\u9009\u62e9\u6700\u4f18\u4e0d\u5e73\u8861\u7b56\u7565\u7684\u6307\u6807"},
    "imb_metric_pr_auc": {"en": "PR-AUC (recommended for imbalance)", "zh": "PR-AUC\uff08\u4e0d\u5e73\u8861\u63a8\u8350\uff09"},
    "imb_metric_roc_auc":{"en": "ROC-AUC", "zh": "ROC-AUC"},

    "c_imbalance":       {"en": "Imbalance:", "zh": "\u4e0d\u5e73\u8861\uff1a"},
    "c_validation":      {"en": "Validation:", "zh": "\u9a8c\u8bc1\u65b9\u5f0f\uff1a"},
    "c_cv_folds":        {"en": "CV Folds:", "zh": "CV \u6298\u6570\uff1a"},
    "c_trials":          {"en": "Tries/model:", "zh": "\u5c1d\u8bd5\u6b21\u6570/\u6a21\u578b\uff1a"},
    "pick_optuna_trials":{"en": "Optuna tries per model (higher = slower, may improve):",
                          "zh": "\u6bcf\u4e2a\u6a21\u578b Optuna \u5c1d\u8bd5\u6b21\u6570\uff08\u8d8a\u5927\u8d8a\u6162\uff0c\u53ef\u80fd\u66f4\u597d\uff09\uff1a"},
    "optuna_trials_hint":{"en": "Quick suggestion: 20 (fast) / 50 (balanced) / 100 (thorough)",
                          "zh": "\u5feb\u901f\u5efa\u8bae\uff1a20\uff08\u5feb\uff09 / 50\uff08\u5e73\u8861\uff09 / 100\uff08\u7ec6\u81f4\uff09"},
    "pick_optuna_trials_preset":{"en": "Pick Optuna trials per model",
                                 "zh": "\u9009\u62e9\u6bcf\u4e2a\u6a21\u578b\u7684 Optuna \u5c1d\u8bd5\u6b21\u6570"},
    "optuna_trials_custom":{"en": "Custom Optuna value...",
                            "zh": "\u81ea\u5b9a\u4e49 Optuna \u6570\u503c..."},
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

def _getch(text_mode: bool = False) -> str:
    """Read one keypress using os.read (unbuffered) to avoid Python stdin buffering.

    When *text_mode* is True, printable characters (including ``q`` and ``a``)
    are returned as-is instead of being mapped to command strings.
    """
    try:
        import tty, termios, select as sel_mod
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd, termios.TCSADRAIN)
            ch = os.read(fd, 1)
            if ch == b"\x1b":
                if not sel_mod.select([fd], [], [], 0.05)[0]:
                    return "ESC"
                ch2 = os.read(fd, 1)
                if ch2 == b"[":
                    ch3 = os.read(fd, 1)
                    if ch3 == b"A": return "UP"
                    if ch3 == b"B": return "DOWN"
                    if ch3 == b"C": return "RIGHT"
                    if ch3 == b"D": return "LEFT"
                    if ch3 in (b"5", b"6"):
                        if sel_mod.select([fd], [], [], 0.05)[0]:
                            ch4 = os.read(fd, 1)
                            if ch4 == b"~":
                                return "PAGE_UP" if ch3 == b"5" else "PAGE_DOWN"
                while sel_mod.select([fd], [], [], 0.02)[0]:
                    os.read(fd, 1)
                return "ESC"
            if ch in (b"\r", b"\n"): return "ENTER"
            if ch == b"\x03": return "CTRL_C"
            if ch == b"\x04": return "CTRL_D"
            if not text_mode:
                if ch == b" ": return "SPACE"
                if ch == b"q": return "Q"
                if ch == b"a": return "A"
            return ch.decode("latin-1")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        raw = input()
        return raw.strip() or "ENTER"


_BACK_SENTINEL = None  # returned by _input_line on ESC


def _input_line(prompt_str: str, default: str = "") -> Optional[str]:
    """Read a line of text using _getch(text_mode=True).

    - **ESC / Ctrl-C / Ctrl-D** → return ``None`` (back signal)
    - **Enter** → return the buffer (or *default* when empty)
    - **Backspace** / **LEFT** / **RIGHT** → editing
    - All printable characters inserted at cursor position
    """
    buf: List[str] = []
    cur = 0
    sys.stdout.write(SHOW_CUR)
    sys.stdout.write(prompt_str)
    sys.stdout.flush()

    def _redraw() -> None:
        sys.stdout.write(f"\r{ERASE}{prompt_str}{''.join(buf)}")
        tail = len(buf) - cur
        if tail > 0:
            sys.stdout.write(f"\033[{tail}D")
        sys.stdout.flush()

    while True:
        key = _getch(text_mode=True)
        if key == "ESC":
            sys.stdout.write(f"\r{ERASE}")
            sys.stdout.flush()
            return None
        elif key in ("CTRL_C", "CTRL_D"):
            sys.stdout.write(f"\r{ERASE}")
            sys.stdout.flush()
            return None
        elif key == "ENTER":
            sys.stdout.write("\n")
            sys.stdout.flush()
            result = "".join(buf).strip()
            return result if result else default
        elif key in ("\x7f", "\x08"):
            if cur > 0:
                buf.pop(cur - 1)
                cur -= 1
                _redraw()
        elif key == "LEFT":
            if cur > 0:
                cur -= 1
                sys.stdout.write("\033[D")
                sys.stdout.flush()
        elif key == "RIGHT":
            if cur < len(buf):
                cur += 1
                sys.stdout.write("\033[C")
                sys.stdout.flush()
        elif key in ("UP", "DOWN", "PAGE_UP", "PAGE_DOWN"):
            pass  # ignore in text input
        elif len(key) == 1 and (key.isprintable() or key == " "):
            buf.insert(cur, key)
            cur += 1
            _redraw()


def _is_back_text_token(raw: str) -> bool:
    token = str(raw or "").strip().lower()
    return token in {"q", "quit", "back", "return", "返回"}


def _notice(message: str) -> None:
    print(f"\n  {s('Y', message)}")
    sys.stdout.write(SHOW_CUR)
    try:
        input(f"  {DIM}{t('enter_continue')}{RST}")
    except (EOFError, KeyboardInterrupt):
        pass


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
    term_h = shutil.get_terminal_size((80, 24)).lines
    max_vis = max(min(n, term_h - 10), 5)
    offset = 0

    def _draw() -> int:
        sys.stdout.write(HIDE_CUR); sys.stdout.flush()
        lc = 0
        if title:
            print(f"  {s('C', title, bold=True)}")
            print()
            lc += 2
        end = min(offset + max_vis, n)
        if offset > 0:
            print(f"  {DIM}  \u25b2 {offset} more{RST}")
            lc += 1
        for i in range(offset, end):
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
            lc += 1
        if end < n:
            print(f"  {DIM}  \u25bc {n - end} more{RST}")
            lc += 1
        print()
        hint = t("nav_first") if is_first else t("nav")
        print(f"  {DIM}{hint}{RST}")
        lc += 2
        return lc

    lc = _draw()
    while True:
        key = _getch()
        if key == "UP" and sel > 0:
            sel -= 1
            if sel < offset:
                offset = sel
        elif key == "DOWN" and sel < n - 1:
            sel += 1
            if sel >= offset + max_vis:
                offset = sel - max_vis + 1
        elif key == "PAGE_UP":
            sel = max(sel - max_vis, 0)
            offset = max(offset - max_vis, 0)
            if sel < offset:
                offset = sel
        elif key == "PAGE_DOWN":
            sel = min(sel + max_vis, n - 1)
            offset = min(offset + max_vis, max(n - max_vis, 0))
            if sel >= offset + max_vis:
                offset = sel - max_vis + 1
        elif key == "ENTER":
            sys.stdout.write(SHOW_CUR); sys.stdout.flush()
            return sel
        elif key in ("Q", "CTRL_C", "CTRL_D", "ESC", "LEFT"):
            sys.stdout.write(SHOW_CUR); sys.stdout.flush()
            return -1
        elif len(key) == 1 and key.isdigit() and 1 <= int(key) <= min(n, 9):
            sel = int(key) - 1
            if sel < offset:
                offset = sel
            elif sel >= offset + max_vis:
                offset = sel - max_vis + 1
        else:
            continue
        for _ in range(lc):
            sys.stdout.write(f"{UP_LINE}{ERASE}")
        sys.stdout.write("\r"); sys.stdout.flush()
        lc = _draw()


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
    term_h = shutil.get_terminal_size((80, 24)).lines
    max_vis = max(min(n, term_h - 10), 5)
    offset = 0

    def _draw() -> int:
        sys.stdout.write(HIDE_CUR); sys.stdout.flush()
        lc = 0
        if title:
            print(f"  {s('C', title, bold=True)}")
            print()
            lc += 2
        end = min(offset + max_vis, n)
        if offset > 0:
            print(f"  {DIM}  \u25b2 {offset} more{RST}")
            lc += 1
        for i in range(offset, end):
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
            lc += 1
        if end < n:
            print(f"  {DIM}  \u25bc {n - end} more{RST}")
            lc += 1
        print()
        print(f"  {DIM}{t('ms_hint')}{RST}")
        lc += 2
        return lc

    lc = _draw()
    while True:
        key = _getch()
        if key == "UP" and sel > 0:
            sel -= 1
            if sel < offset:
                offset = sel
        elif key == "DOWN" and sel < n - 1:
            sel += 1
            if sel >= offset + max_vis:
                offset = sel - max_vis + 1
        elif key == "PAGE_UP":
            sel = max(sel - max_vis, 0)
            offset = max(offset - max_vis, 0)
            if sel < offset:
                offset = sel
        elif key == "PAGE_DOWN":
            sel = min(sel + max_vis, n - 1)
            offset = min(offset + max_vis, max(n - max_vis, 0))
            if sel >= offset + max_vis:
                offset = sel - max_vis + 1
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
        elif key in ("Q", "CTRL_C", "CTRL_D", "ESC", "LEFT"):
            sys.stdout.write(SHOW_CUR); sys.stdout.flush()
            return None
        else:
            continue
        for _ in range(lc):
            sys.stdout.write(f"{UP_LINE}{ERASE}")
        sys.stdout.write("\r"); sys.stdout.flush()
        lc = _draw()


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
        if self._t: self._t.join(timeout=2)
        sys.stdout.write(f"\r{ERASE}{SHOW_CUR}"); sys.stdout.flush()
    def _run(self) -> None:
        for f in itertools.cycle(self.FRAMES):
            if self._stop.is_set(): break
            sys.stdout.write(f"\r  {s('C', f)} {s('W', self.label)}")
            sys.stdout.flush()
            self._stop.wait(0.08)


def run_spinner(cmd: List[str], label: str, cwd: str = "",
                timeout: int = 1800) -> Tuple[int, str, str]:
    with Spinner(label):
        try:
            p = subprocess.run(cmd, cwd=cwd or str(REPO_ROOT),
                               capture_output=True, text=True,
                               timeout=timeout,
                               start_new_session=True)
        except subprocess.TimeoutExpired as exc:
            return (
                124,
                exc.stdout or "",
                (exc.stderr or "") + f"\n[TIMEOUT] Process killed after {timeout}s.\n",
            )
        except KeyboardInterrupt:
            return (130, "", "\n[INTERRUPTED] Process terminated by user.\n")
    return p.returncode, p.stdout, p.stderr


def run_with_progress(cmd: List[str], label: str, total: int = 0,
                      cwd: str = "", timeout: int = 3600) -> Tuple[int, str, str]:
    """Run a subprocess while parsing ``[PROGRESS] i/n model_id`` lines from stderr.

    Displays a real-time progress bar with the current model name.
    Falls back to a spinner if no [PROGRESS] lines arrive.
    """
    import re as _re
    _prog_re = _re.compile(r"\[PROGRESS\]\s+(\d+)/(\d+)\s+(.*)")
    bar_w = 24
    current = 0
    cur_model = ""
    stderr_lines: List[str] = []
    stdout_buf: List[str] = []

    sys.stdout.write(HIDE_CUR)
    sys.stdout.flush()

    def _draw_bar(done: int, total_n: int, model_name: str) -> None:
        if total_n <= 0:
            total_n = 1
        pct = min(int(done * 100 / total_n), 100)
        filled = int(bar_w * done / total_n)
        bar = s('C', '\u2588' * filled) + DIM + '\u2591' * (bar_w - filled) + RST
        name = model_name[:30] if model_name else ""
        sys.stdout.write(f"\r{ERASE}  {bar} {s('W', f'{pct:>3}%')}  "
                         f"{s('W', label)} {s('D', name)}")
        sys.stdout.flush()

    _draw_bar(0, max(total, 1), "")

    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd or str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        import selectors as _sel
        sel = _sel.DefaultSelector()
        sel.register(proc.stdout, _sel.EVENT_READ)
        sel.register(proc.stderr, _sel.EVENT_READ)

        open_streams = 2
        while open_streams > 0:
            events = sel.select(timeout=0.2)
            if not events:
                # animate spinner in the model name field while waiting
                spin_frames = Spinner.FRAMES
                idx = int(time.time() * 10) % len(spin_frames)
                _draw_bar(current, max(total, 1),
                          cur_model + " " + spin_frames[idx] if cur_model else spin_frames[idx])
                continue
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    sel.unregister(key.fileobj)
                    open_streams -= 1
                    continue
                if key.fileobj is proc.stderr:
                    stderr_lines.append(line)
                    m = _prog_re.match(line.strip())
                    if m:
                        current = int(m.group(1))
                        total = int(m.group(2))
                        cur_model = m.group(3).strip()
                        _draw_bar(current, total, cur_model)
                else:
                    stdout_buf.append(line)

        proc.wait(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = 124
        stderr_lines.append(f"\n[TIMEOUT] Process killed after {timeout}s.\n")
    except KeyboardInterrupt:
        proc.kill()
        rc = 130
        stderr_lines.append("\n[INTERRUPTED] Process terminated by user.\n")

    # final complete bar
    if total > 0:
        _draw_bar(total, total, s('G', '\u2713'))
    sys.stdout.write(f"\r{ERASE}{SHOW_CUR}")
    sys.stdout.flush()

    return rc, "".join(stdout_buf), "".join(stderr_lines)


# ══════════════════════════════════════════════════════════════════════════════
#  SMART DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_PID_HINTS = ["patient_id", "patientid", "patient", "subject_id", "subjectid",
              "sample_id", "pid", "mrn", "record_id", "case_id"]
_TGT_HINTS = ["target", "label", "y", "outcome", "diagnosis", "class",
              "result", "status", "disease", "mortality"]
_TIME_HINTS = ["time", "date", "timestamp", "event_time", "datetime",
               "admission", "visit_date", "created_at"]


def _hint_match(hint: str, col_lower: str) -> bool:
    if "_" in hint:
        return hint in col_lower
    return hint in col_lower.split("_")


def detect_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    pid = target = time_col = None
    for col in cols:
        low = col.lower().strip().replace(" ", "_")
        if not pid:
            for h in _PID_HINTS:
                if _hint_match(h, low): pid = col; break
        if not target:
            for h in _TGT_HINTS:
                if _hint_match(h, low): target = col; break
        if not time_col:
            for h in _TIME_HINTS:
                if _hint_match(h, low): time_col = col; break
    return {"pid": pid, "target": target, "time": time_col}


def scan_csv() -> List[Path]:
    _MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    _DIR_TIMEOUT = 2.0  # seconds per directory
    _MAX_TOTAL = 30
    found: List[Path] = []
    seen: set[Path] = set()
    for d in [EXAMPLES_DIR, DESKTOP, Path.home()/"Downloads",
              Path.home()/"Documents", REPO_ROOT, DEFAULT_OUT]:
        if len(found) >= _MAX_TOTAL:
            break
        if d.is_dir():
            try:
                t0 = time.time()
                for f in sorted(d.glob("*.csv"))[:10]:
                    if time.time() - t0 > _DIR_TIMEOUT:
                        break
                    try:
                        st = f.stat()
                        if f not in seen and 100 < st.st_size <= _MAX_FILE_SIZE:
                            found.append(f)
                            seen.add(f)
                    except (PermissionError, OSError):
                        pass
                    if len(found) >= _MAX_TOTAL:
                        break
            except (PermissionError, OSError):
                pass
    return found[:_MAX_TOTAL]


def csv_cols(path: Path) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return [c.strip() for c in next(csv.reader(fh), []) if c.strip()]
    except Exception:
        return []


def csv_rows(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)
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
    ("lightgbm",                  "m_lgbm"),
    ("tabpfn",                    "m_tabpfn"),
]
OPTIONAL_MODEL_MODULES = {
    "xgboost": "xgboost",
    "catboost": "catboost",
    "lightgbm": "lightgbm",
    "tabpfn": "tabpfn",
}
OPTIONAL_MODEL_INSTALL_PACKAGES = {
    "xgboost": "xgboost",
    "catboost": "catboost",
    "lightgbm": "lightgbm",
    "tabpfn": "tabpfn",
}
# Conservative default for clinical small/medium datasets:
# linear models are usually more stable and easier to calibrate.
DEFAULT_MODELS = [0, 1, 2]  # L1, L2, ElasticNet
STRICT_SMALL_SAMPLE_DEFAULT_MAX_ROWS = 1200
STRICT_SMALL_SAMPLE_MAX_TRIALS_CAP = 8
STRICT_SMALL_SAMPLE_MODEL_POOL = ["logistic_l1", "logistic_l2", "logistic_elasticnet"]
PLAY_DOWNLOAD_DATASETS = [
    ("heart", "heart_disease", "ds_heart", "ds_heart_d"),
    ("breast", "breast_cancer", "ds_breast", "ds_breast_d"),
    ("ckd", "chronic_kidney_disease", "ds_kidney", "ds_kidney_d"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  RUN HISTORY
# ══════════════════════════════════════════════════════════════════════════════

_HISTORY_PATH = Path.home() / ".mlgg" / "history.json"

_HISTORY_KEYS = [
    "source", "dataset_key", "csv_path", "pid", "target", "time",
    "strategy", "train_ratio", "valid_ratio", "test_ratio",
    "validation_method", "cv_folds", "imbalance_strategy", "imbalance_strategies", "imbalance_selection_metric",
    "model_pool", "hyperparam_search", "calibration", "device",
    "out_dir", "ignore_cols", "n_jobs", "max_trials",
    "optuna_trials", "include_optional_models",
]


def recommended_max_trials(state: Dict[str, Any]) -> int:
    """Heuristic default for hyperparameter trials to reduce small-sample variance."""
    search = str(state.get("hyperparam_search", "fixed_grid")).strip().lower()
    try:
        n_rows = int(state.get("_n_rows", 0) or 0)
    except Exception:
        n_rows = 0
    if search == "fixed_grid":
        return 1
    if search == "optuna":
        if 0 < n_rows <= 500:
            return 20
        if 500 < n_rows <= 1500:
            return 30
        return 50
    # random_subsample
    if 0 < n_rows <= 500:
        return 8
    if 500 < n_rows <= 1500:
        return 12
    return 20


def state_n_rows(state: Dict[str, Any]) -> int:
    """Best-effort row count resolution used for small-sample safeguards."""
    for key in ("_n_rows", "_rows"):
        try:
            value = int(state.get(key, 0) or 0)
            if value > 0:
                return value
        except Exception:
            pass
    csv_path = str(state.get("csv_path", "") or "").strip()
    if csv_path and Path(csv_path).exists():
        try:
            return int(csv_rows(Path(csv_path)))
        except Exception:
            return 0
    return 0


def available_columns_for_ignore(state: Dict[str, Any]) -> List[str]:
    """Best-effort detected columns for advanced ignore-cols editing."""
    cols = state.get("_columns")
    if isinstance(cols, list):
        out = [str(c).strip() for c in cols if str(c).strip()]
        if out:
            return out
    csv_path = str(state.get("csv_path", "") or "").strip()
    if csv_path and Path(csv_path).exists():
        try:
            out = [str(c).strip() for c in csv_cols(Path(csv_path)) if str(c).strip()]
            if out:
                return out
        except Exception:
            pass
    return []


def normalize_ignore_columns(state: Dict[str, Any], tokens: List[str]) -> str:
    """
    Normalize ignore columns:
    - trim whitespace
    - de-duplicate while preserving order
    - force patient_id/time columns to stay ignored
    """
    ordered: List[str] = []
    seen = set()
    for token in tokens:
        val = str(token).strip()
        if val and val not in seen:
            ordered.append(val)
            seen.add(val)
    mandatory = [str(state.get("pid", "")).strip(), str(state.get("time", "")).strip()]
    for col in mandatory:
        if col and col not in seen:
            ordered.append(col)
            seen.add(col)
    return ",".join(ordered)


def optional_backend_available(model_family: str) -> bool:
    """Return True when an optional model backend appears installed."""
    module_name = OPTIONAL_MODEL_MODULES.get(str(model_family).strip())
    if not module_name:
        return True
    try:
        importlib.invalidate_caches()
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def optuna_backend_available() -> bool:
    """Return True when optuna can be imported in current runtime."""
    try:
        importlib.invalidate_caches()
        return importlib.util.find_spec("optuna") is not None
    except Exception:
        return False


def model_pool_tokens_from_state(state: Dict[str, Any]) -> List[str]:
    return [token.strip() for token in str(state.get("model_pool", "")).split(",") if token.strip()]


def apply_model_pool_tokens(state: Dict[str, Any], tokens: List[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for token in tokens:
        name = str(token).strip()
        if name and name not in seen:
            ordered.append(name)
            seen.add(name)
    if not ordered:
        ordered = ["logistic_l2"]
    state["model_pool"] = ",".join(ordered)
    label_map = {name: t(label_key) for name, label_key in MODEL_POOL}
    state["_model_labels"] = [label_map.get(name, name) for name in ordered]
    return ordered


def enforce_optional_backend_policy(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply include_optional_models flag to the already selected model pool."""
    current = model_pool_tokens_from_state(state)
    include_optional = bool(state.get("include_optional_models", False))
    if include_optional:
        has_optional = any(family in OPTIONAL_MODEL_MODULES for family in current)
        if not has_optional:
            state["include_optional_models"] = False
        return {"changed": False, "removed": [], "kept": current, "fallback_used": False}

    kept = [family for family in current if family not in OPTIONAL_MODEL_MODULES]
    removed = [family for family in current if family in OPTIONAL_MODEL_MODULES]
    fallback_used = False
    if not kept:
        kept = ["logistic_l2"]
        fallback_used = True
    changed = kept != current
    if changed:
        apply_model_pool_tokens(state, kept)
    return {"changed": changed, "removed": removed, "kept": kept, "fallback_used": fallback_used}


def normalize_optional_backend_state(state: Dict[str, Any]) -> None:
    """
    Ensure optional-backend flag and model_pool stay consistent.

    Important for history replay paths where advanced/model steps are skipped.
    """
    source = str(state.get("source", "")).strip().lower()
    if source not in {"download", "csv"}:
        return
    if "model_pool" not in state:
        return
    enforce_optional_backend_policy(state)


def prune_unavailable_optional_models(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove optional model families whose backend is unavailable.

    If all selected models are pruned, fall back to logistic_l2 to keep play flow usable.
    """
    raw_pool = model_pool_tokens_from_state(state)
    kept: List[str] = []
    removed: List[str] = []
    for family in raw_pool:
        if family in OPTIONAL_MODEL_MODULES and not optional_backend_available(family):
            removed.append(family)
            continue
        kept.append(family)

    fallback_used = False
    if not kept:
        kept = ["logistic_l2"]
        fallback_used = True

    changed = kept != raw_pool
    if changed:
        apply_model_pool_tokens(state, kept)
    return {
        "changed": changed,
        "removed": removed,
        "kept": kept,
        "fallback_used": fallback_used,
    }


def collect_runtime_dependency_issues(state: Dict[str, Any]) -> Dict[str, Any]:
    """Collect missing runtime dependencies implied by current wizard selections."""
    raw_pool = model_pool_tokens_from_state(state)
    missing_optional: List[str] = []
    for family in raw_pool:
        if family in OPTIONAL_MODEL_MODULES and not optional_backend_available(family):
            if family not in missing_optional:
                missing_optional.append(family)
    need_optuna = str(state.get("hyperparam_search", "")).strip().lower() == "optuna"
    optuna_missing = bool(need_optuna and not optuna_backend_available())
    return {
        "missing_optional": missing_optional,
        "optuna_missing": optuna_missing,
        "has_issues": bool(missing_optional or optuna_missing),
    }


def apply_dependency_downgrade(state: Dict[str, Any], issues: Dict[str, Any]) -> Dict[str, Any]:
    """Apply safe fallback when user chooses not to install missing dependencies."""
    before_pool = model_pool_tokens_from_state(state)
    missing_optional = [str(x) for x in issues.get("missing_optional", []) if str(x).strip()]
    optuna_missing = bool(issues.get("optuna_missing", False))
    removed: List[str] = []
    fallback_used = False
    if missing_optional:
        prune_result = prune_unavailable_optional_models(state)
        removed = [str(x) for x in prune_result.get("removed", []) if str(x).strip()]
        fallback_used = bool(prune_result.get("fallback_used", False))
    downgraded_optuna = False
    if optuna_missing and str(state.get("hyperparam_search", "")).strip().lower() == "optuna":
        state["hyperparam_search"] = "random_subsample"
        try:
            current_trials = int(state.get("max_trials", 0) or 0)
        except Exception:
            current_trials = 0
        if current_trials < 1:
            state["max_trials"] = int(recommended_max_trials(state))
        downgraded_optuna = True
    kept = model_pool_tokens_from_state(state)
    if not fallback_used and kept == ["logistic_l2"] and before_pool != ["logistic_l2"]:
        fallback_used = True
    return {
        "removed_optional": removed,
        "downgraded_optuna": downgraded_optuna,
        "kept_model_pool": kept,
        "fallback_used": fallback_used,
    }


def _print_dependency_downgrade_summary(downgrade: Dict[str, Any]) -> None:
    removed = [str(x) for x in downgrade.get("removed_optional", []) if str(x).strip()]
    if removed:
        print(f"\n  {s('Y', t('dep_downgrade_optional_removed'))} {', '.join(removed)}")
    if bool(downgrade.get("downgraded_optuna", False)):
        print(f"  {s('Y', t('dep_downgrade_optuna'))}")
    kept = [str(x) for x in downgrade.get("kept_model_pool", []) if str(x).strip()]
    if kept:
        print(f"  {s('C', t('dep_downgrade_model_pool'))} {', '.join(kept)}")
    if bool(downgrade.get("fallback_used", False)):
        print(f"  {s('C', t('dep_downgrade_fallback'))}")
    print()


def _install_dependency_packages(packages: List[str]) -> Dict[str, Any]:
    installed: List[str] = []
    failed: List[Dict[str, Any]] = []
    for pkg in packages:
        pkg_name = str(pkg).strip()
        if not pkg_name:
            continue
        rc, _, err = run_spinner(
            [sys.executable, "-m", "pip", "install", pkg_name],
            t("dep_installing_pkg", pkg=pkg_name),
        )
        if rc == 0:
            installed.append(pkg_name)
            continue
        tail = [line for line in str(err).strip().split("\n") if line][-3:]
        failed.append({"package": pkg_name, "stderr_tail": tail})
    return {"installed": installed, "failed": failed}


def ensure_runtime_dependencies(state: Dict[str, Any]) -> bool:
    """
    Interactive runtime dependency resolver for play mode.
    Returns True when resolved; False when user cancels or issues remain unresolved.
    """
    while True:
        issues = collect_runtime_dependency_issues(state)
        if not issues.get("has_issues", False):
            return True

        lines: List[str] = []
        missing_optional = [str(x) for x in issues.get("missing_optional", []) if str(x).strip()]
        optuna_missing = bool(issues.get("optuna_missing", False))
        if missing_optional:
            lines.append(f"{t('dep_missing_optional')} {', '.join(missing_optional)}")
            lines.append("")
        if optuna_missing:
            lines.append(t("dep_missing_optuna"))
            lines.append("")
        install_targets: List[str] = []
        for family in missing_optional:
            pkg = OPTIONAL_MODEL_INSTALL_PACKAGES.get(family)
            if pkg and pkg not in install_targets:
                install_targets.append(pkg)
        if optuna_missing and "optuna" not in install_targets:
            install_targets.append("optuna")
        if install_targets:
            lines.append(f"{t('dep_install_cmd')} {sys.executable} -m pip install {' '.join(install_targets)}")
        box(t("dep_fix_title"), lines, color="Y")

        choice = select(
            [t("dep_action_install"), t("dep_action_downgrade"), t("dep_action_cancel")],
            title=t("dep_fix_title"),
        )
        if choice < 0 or choice == 2:
            return False
        if choice == 0:
            pending_targets = list(install_targets)
            while pending_targets:
                install_result = _install_dependency_packages(pending_targets)
                ok_pkgs = [str(x) for x in install_result.get("installed", []) if str(x).strip()]
                failed_items = [
                    item for item in install_result.get("failed", [])
                    if isinstance(item, dict) and str(item.get("package", "")).strip()
                ]
                if ok_pkgs:
                    print(f"\n  {s('G', t('dep_install_ok_pkgs'))} {', '.join(ok_pkgs)}")
                if not failed_items:
                    print(f"  {s('G', t('dep_install_success'))}\n")
                    break

                fail_lines = [t("dep_install_partial"), "", f"{t('dep_install_failed_pkgs')} {', '.join(str(item.get('package')) for item in failed_items)}"]
                for item in failed_items:
                    stderr_tail = [str(line).strip() for line in item.get("stderr_tail", []) if str(line).strip()]
                    if stderr_tail:
                        fail_lines.append(f"- {item.get('package')}: {stderr_tail[-1]}")
                box(t("dep_install_failed"), fail_lines, color="R")

                next_choice = select(
                    [t("dep_action_retry_failed"), t("dep_action_downgrade"), t("dep_action_cancel")],
                    title=t("dep_fix_title"),
                )
                if next_choice < 0 or next_choice == 2:
                    return False
                if next_choice == 1:
                    downgrade = apply_dependency_downgrade(state, collect_runtime_dependency_issues(state))
                    _print_dependency_downgrade_summary(downgrade)
                    return True
                pending_targets = [str(item.get("package")).strip() for item in failed_items if str(item.get("package", "")).strip()]
            continue

        downgrade = apply_dependency_downgrade(state, issues)
        _print_dependency_downgrade_summary(downgrade)
        return True


def strict_small_sample_active(state: Dict[str, Any]) -> bool:
    if not bool(state.get("_strict_small_sample", False)):
        return False
    try:
        max_rows = int(state.get("_strict_small_sample_max_rows", STRICT_SMALL_SAMPLE_DEFAULT_MAX_ROWS) or 0)
    except Exception:
        max_rows = STRICT_SMALL_SAMPLE_DEFAULT_MAX_ROWS
    n_rows = state_n_rows(state)
    return bool(max_rows > 0 and n_rows > 0 and n_rows <= max_rows)


def apply_strict_small_sample_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce conservative training defaults for small datasets.
    Returns a small audit payload of applied changes.
    """
    applied: List[str] = []
    n_rows = state_n_rows(state)
    if not strict_small_sample_active(state):
        return {"active": False, "n_rows": n_rows, "applied": applied}

    # 1) Restrict to linear regularized models.
    current_models = [token.strip() for token in str(state.get("model_pool", "")).split(",") if token.strip()]
    filtered_models = [m for m in current_models if m in STRICT_SMALL_SAMPLE_MODEL_POOL]
    if not filtered_models:
        filtered_models = list(STRICT_SMALL_SAMPLE_MODEL_POOL)
    new_pool = ",".join(filtered_models)
    if new_pool != state.get("model_pool"):
        state["model_pool"] = new_pool
        applied.append("model_pool_linear_only")

    # 2) Cap search complexity.
    if str(state.get("hyperparam_search", "")).strip().lower() == "optuna":
        state["hyperparam_search"] = "random_subsample"
        applied.append("disable_optuna")

    rec_trials = int(recommended_max_trials(state))
    capped_trials = min(STRICT_SMALL_SAMPLE_MAX_TRIALS_CAP, rec_trials)
    try:
        current_trials = int(state.get("max_trials", capped_trials) or capped_trials)
    except Exception:
        current_trials = capped_trials
    new_trials = min(current_trials, capped_trials)
    if int(state.get("max_trials", new_trials) or new_trials) != new_trials:
        applied.append("cap_max_trials")
    state["max_trials"] = int(new_trials)

    # 3) Keep calibration conservative.
    if str(state.get("calibration", "none")).strip().lower() not in {"none", "power"}:
        state["calibration"] = "power"
        applied.append("calibration_to_power")

    return {
        "active": True,
        "n_rows": n_rows,
        "max_rows": int(state.get("_strict_small_sample_max_rows", STRICT_SMALL_SAMPLE_DEFAULT_MAX_ROWS)),
        "applied": applied,
    }


def split_strategy_order_for_source(source: str) -> List[str]:
    """Return split strategy order with source-aware default at index 0."""
    token = str(source).strip().lower()
    if token in {"download", "csv"}:
        # UCI packaged examples include synthetic event_time; default to
        # prevalence-stable grouped split unless users explicitly need temporal.
        return ["stratified_grouped", "grouped_temporal"]
    return ["grouped_temporal", "stratified_grouped"]


def _save_history(state: Dict) -> None:
    """Save current run config to ~/.mlgg/history.json."""
    try:
        import json as _json, datetime as _dt
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {k: state[k] for k in _HISTORY_KEYS if k in state}
        snapshot["_timestamp"] = _dt.datetime.now().isoformat()
        _HISTORY_PATH.write_text(_json.dumps(snapshot, indent=2, ensure_ascii=False))
    except Exception:
        pass


def _load_history() -> Optional[Dict]:
    """Load last run config from ~/.mlgg/history.json. Returns None if unavailable."""
    try:
        import json as _json
        if _HISTORY_PATH.exists():
            data = _json.loads(_HISTORY_PATH.read_text())
            if isinstance(data, dict) and "source" in data:
                return data
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  WIZARD STEPS
# ══════════════════════════════════════════════════════════════════════════════

def prompt(label: str, default: str = "") -> Optional[str]:
    """Prompt the user for a text input with optional default value.

    Returns None when the user presses ESC (back signal).
    Returns "" only on empty input with no default.
    """
    suffix = f" [{default}]" if default else ""
    prompt_str = f"  {s('C','>')} {s('W', label)}{suffix}: "
    result = _input_line(prompt_str, default="")
    if result is None:
        return None
    return result if result else default


def step_lang(state: Dict) -> Any:
    _clear()
    if not _TEST_MODE:
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
    opts = [t("src_download"), t("src_csv")]
    descs = [t("src_download_d"), t("src_csv_d")]
    ci = select(opts, descs)
    if ci < 0:
        return BACK
    state["source"] = ["download", "csv"][ci]
    return True


def step_dataset(state: Dict) -> Any:
    if state.get("_from_history"):
        return SKIP
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
        state["_n_rows"] = 0
        return SKIP

    if source == "full":
        _clear()
        step_header(3, TOTAL_STEPS, t("s_dataset"))
        print(f"\n  {s('W', t('src_full'))}")
        print(f"  {s('D', t('src_full_d'))}\n")
        project_root = prompt("Project root (with request.json)" if LANG == "en" else "\u9879\u76ee\u76ee\u5f55\uff08\u542b request.json\uff09")
        if not project_root:
            return BACK
        p = Path(project_root).expanduser().resolve()
        req = p / "evidence" / "request.json"
        if not req.exists():
            req = p / "request.json"
        if not req.exists():
            missing_msg = "request.json not found in" if LANG == "en" else "\u672a\u5728\u4ee5\u4e0b\u76ee\u5f55\u627e\u5230 request.json\uff1a"
            hint_msg = "Run mlgg.py onboarding first to generate it." if LANG == "en" else "\u8bf7\u5148\u8fd0\u884c mlgg.py onboarding \u751f\u6210\u8be5\u6587\u4ef6\u3002"
            back_msg = "Press Enter to go back" if LANG == "en" else "\u6309 Enter \u8fd4\u56de"
            print(f"\n  {s('R', missing_msg)} {p}")
            print(f"  {s('D', hint_msg)}\n")
            prompt(back_msg)
            return BACK
        state["_full_project_root"] = str(p)
        state["_full_request_json"] = str(req)
        state["out_dir"] = str(p)
        state["dataset_key"] = "full_pipeline"
        return SKIP

    _clear()
    step_header(3, TOTAL_STEPS, t("s_dataset"))

    if source == "download":
        labels = [t(label_key) for _, _, label_key, _ in PLAY_DOWNLOAD_DATASETS]
        descs = [t(desc_key) for _, _, _, desc_key in PLAY_DOWNLOAD_DATASETS]
        ci = select(labels, descs)
        if ci < 0:
            return BACK
        key, file_stem, _, _ = PLAY_DOWNLOAD_DATASETS[ci]
        state["dataset_key"] = key
        state["dataset_file"] = file_stem
        state["csv_path"] = str(EXAMPLES_DIR / f"{file_stem}.csv")
        try:
            state["_n_rows"] = int(csv_rows(Path(state["csv_path"])))
        except Exception:
            state["_n_rows"] = 0
        state["out_dir"] = str(DEFAULT_OUT / file_stem)
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
            path = _input_line(f"  {s('C','>')} {s('W', t('csv_prompt'))}: ")
            if path is None:
                return BACK
        else:
            path = str(files[fi])
    else:
        path = _input_line(f"  {s('C','>')} {s('W', t('csv_prompt'))}: ")
        if path is None:
            return BACK

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
    try:
        state["_n_rows"] = int(csv_rows(Path(path)))
    except Exception:
        state["_n_rows"] = 0
    state["out_dir"] = str(DEFAULT_OUT / Path(path).stem)
    return True


def step_config(state: Dict) -> Any:
    """Column selection -- Patient ID + Target (CSV mode only)."""
    if state.get("_from_history"):
        return SKIP
    if state.get("source") in ("demo", "download", "full"):
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

    if len(columns) < 2:
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

    sub = 0
    pid = tgt = ""
    while True:
        if sub == 0:
            _config_header()
            pid_opts = columns[:]
            if detected["pid"] and detected["pid"] in pid_opts:
                pid_opts.remove(detected["pid"])
                pid_opts.insert(0, detected["pid"])
            pi = select(pid_opts, title=t("pick_pid"))
            if pi < 0: return BACK
            pid = pid_opts[pi]
            sub = 1
        elif sub == 1:
            _config_header((t("c_pid"), pid))
            rem1 = [c for c in columns if c != pid]
            tgt_opts = rem1[:]
            if detected["target"] and detected["target"] in tgt_opts:
                tgt_opts.remove(detected["target"])
                tgt_opts.insert(0, detected["target"])
            ti = select(tgt_opts, title=t("pick_target"))
            if ti < 0:
                sub = 0; continue
            tgt = tgt_opts[ti]
            break

    state["pid"] = pid
    state["target"] = tgt
    return True


def step_split(state: Dict) -> Any:
    """Strategy + validation method + time column (if temporal) + split ratio."""
    if state.get("_from_history"):
        return SKIP
    if state.get("source") in ("demo", "full"):
        return SKIP

    source = state["source"]
    sub = 0
    strat = tcol = ""

    while True:
        if sub == 0:
            _clear()
            step_header(5, TOTAL_STEPS, t("s_split"))
            print(f"  {s('C', t('split_help_title'), bold=True)}")
            print(f"  {DIM}- {t('split_help_temporal')}{RST}")
            print(f"  {DIM}- {t('split_help_stratified')}{RST}")
            if source == "csv":
                print(f"  {DIM}{t('split_help_default_csv')}{RST}\n")
            elif source == "download":
                print(f"  {DIM}{t('split_help_default_download')}{RST}\n")
            else:
                print()
            strategy_order = split_strategy_order_for_source(source)
            strategy_title = {
                "grouped_temporal": t("strat_temporal"),
                "stratified_grouped": t("strat_stratified"),
            }
            strategy_desc = {
                "grouped_temporal": t("strat_temporal_d"),
                "stratified_grouped": t("strat_stratified_d"),
            }
            si = select(
                [strategy_title[item] for item in strategy_order],
                [strategy_desc[item] for item in strategy_order],
                title=t("pick_strat"),
            )
            if si < 0: return BACK
            strat = strategy_order[si]
            state["strategy"] = strat
            tcol = ""
            if strat == "grouped_temporal" and source == "csv":
                sub = 1
            elif strat == "grouped_temporal":
                tcol = state.get("time", "event_time")
                sub = 2
            else:
                sub = 2

        elif sub == 1:
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
                sub = 0; continue
            _clear()
            step_header(5, TOTAL_STEPS, t("s_split"))
            print(f"  {s('G', '\u2713')} {t('c_strat')} {s('W', strat)}\n")
            time_opts = rem[:]
            if detected.get("time") and detected["time"] in time_opts:
                time_opts.remove(detected["time"])
                time_opts.insert(0, detected["time"])
            tci = select(time_opts, title=t("pick_time"))
            if tci < 0:
                sub = 0; continue
            tcol = time_opts[tci]
            sub = 2

        elif sub == 2:
            _clear()
            step_header(5, TOTAL_STEPS, t("s_split"))
            print(f"  {s('G', '\u2713')} {t('c_strat')} {s('W', strat)}")
            if tcol:
                print(f"  {s('G', '\u2713')} {t('c_time')} {s('W', tcol)}")
            print()
            vi = select(
                [t("valid_holdout"), t("valid_cv")],
                [t("valid_holdout_d"), t("valid_cv_d")],
                title=t("pick_valid_method"),
            )
            if vi < 0:
                sub = 1 if (strat == "grouped_temporal" and source == "csv") else 0
                continue
            state["validation_method"] = "holdout" if vi == 0 else "cv"
            if vi == 1:
                sub = 3
            else:
                sub = 4

        elif sub == 3:
            _clear()
            step_header(5, TOTAL_STEPS, t("s_split"))
            print(f"  {s('G', '\u2713')} {t('c_strat')} {s('W', strat)}")
            if tcol:
                print(f"  {s('G', '\u2713')} {t('c_time')} {s('W', tcol)}")
            print(f"  {s('G', '\u2713')} {t('c_validation')} {s('W', t('valid_cv'))}")
            print()
            fi = select(
                [t("cv_3"), t("cv_5"), t("cv_10")],
                title=t("pick_cv_folds"),
            )
            if fi < 0:
                sub = 2; continue
            state["cv_folds"] = [3, 5, 10][fi]
            sub = 5

        elif sub == 4:
            _clear()
            step_header(5, TOTAL_STEPS, t("s_split"))
            print(f"  {s('G', '\u2713')} {t('c_strat')} {s('W', strat)}")
            if tcol:
                print(f"  {s('G', '\u2713')} {t('c_time')} {s('W', tcol)}")
            print(f"  {s('G', '\u2713')} {t('c_validation')} {s('W', t('valid_holdout'))}")
            print()
            ri = select(
                [t("ratio_60"), t("ratio_70"), t("ratio_70_20_10"), t("ratio_80")],
                title=t("pick_ratio"),
            )
            if ri < 0:
                sub = 2; continue
            ratios = [(0.6, 0.2, 0.2), (0.7, 0.15, 0.15), (0.7, 0.2, 0.1), (0.8, 0.1, 0.1)]
            state["train_ratio"], state["valid_ratio"], state["test_ratio"] = ratios[ri]
            state.setdefault("cv_folds", 5)
            break

        elif sub == 5:
            _clear()
            step_header(5, TOTAL_STEPS, t("s_split"))
            print(f"  {s('G', '\u2713')} {t('c_strat')} {s('W', strat)}")
            if tcol:
                print(f"  {s('G', '\u2713')} {t('c_time')} {s('W', tcol)}")
            cv_label = t("valid_cv")
            folds = state["cv_folds"]
            print(f"  {s('G', '\u2713')} {t('c_validation')} {s('W', f'{cv_label} ({folds}-fold)')}")
            print()
            ri = select(
                [t("tt_70_30"), t("tt_80_20")],
                title=t("pick_tt_ratio"),
            )
            if ri < 0:
                sub = 3; continue
            if ri == 0:
                state["train_ratio"], state["valid_ratio"], state["test_ratio"] = 0.6, 0.1, 0.3
            else:
                state["train_ratio"], state["valid_ratio"], state["test_ratio"] = 0.7, 0.1, 0.2
            break

    state["time"] = tcol
    return True


def step_imbalance(state: Dict) -> Any:
    """Class imbalance handling strategy."""
    if state.get("source") in ("demo", "full") or state.get("_from_history"):
        return SKIP

    _clear()
    step_header(6, TOTAL_STEPS, t("s_imbalance"))

    selected = multi_select(
        [t("imb_auto"), t("imb_none"), t("imb_weight"),
         t("imb_smote"), t("imb_ros"), t("imb_rus"), t("imb_adasyn")],
        [t("imb_auto_d"), t("imb_none_d"), t("imb_weight_d"),
         t("imb_smote_d"), t("imb_ros_d"), t("imb_rus_d"), t("imb_adasyn_d")],
        title=t("pick_imbalance"),
        defaults=[0],
    )
    if selected is None:
        return BACK
    if not selected:
        selected = [0]
    strategies = ["auto", "none", "class_weight", "smote",
                  "random_oversample", "random_undersample", "adasyn"]
    chosen = [strategies[i] for i in selected if 0 <= i < len(strategies)]
    if not chosen:
        chosen = ["auto"]
    state["imbalance_strategies"] = chosen
    state["imbalance_strategy"] = chosen[0]

    if len(chosen) > 1:
        _clear()
        step_header(6, TOTAL_STEPS, t("s_imbalance"))
        metric_idx = select(
            [t("imb_metric_pr_auc"), t("imb_metric_roc_auc")],
            title=t("pick_imb_metric"),
        )
        if metric_idx < 0:
            return BACK
        state["imbalance_selection_metric"] = "pr_auc" if metric_idx == 0 else "roc_auc"
    else:
        state["imbalance_selection_metric"] = "pr_auc"
    return True


def step_models(state: Dict) -> Any:
    """Multi-select model families."""
    if state.get("source") in ("demo", "full") or state.get("_from_history"):
        return SKIP

    _clear()
    step_header(7, TOTAL_STEPS, t("s_models"))
    if strict_small_sample_active(state):
        if LANG == "zh":
            print(f"  {s('C', '小样本严格模式：训练前将仅保留线性正则模型')}\n")
        else:
            print(f"  {s('C', 'Strict small-sample mode: non-linear models will be filtered before training')}\n")

    labels: List[str] = []
    canonical_labels: List[str] = []
    for family, key in MODEL_POOL:
        base = t(key)
        canonical_labels.append(base)
        if family in OPTIONAL_MODEL_MODULES:
            if optional_backend_available(family):
                suffix = " (installed)" if LANG == "en" else "（已安装）"
            else:
                suffix = " (not installed)" if LANG == "en" else "（未安装）"
            labels.append(base + suffix)
        else:
            labels.append(base)
    selected = multi_select(labels, title=t("pick_models"), defaults=DEFAULT_MODELS)
    if selected is None:
        return BACK
    if not selected:
        selected = list(DEFAULT_MODELS)
    state["model_pool"] = ",".join(MODEL_POOL[i][0] for i in selected)
    state["_model_labels"] = [canonical_labels[i] for i in selected]
    state["include_optional_models"] = any(
        MODEL_POOL[i][0] in OPTIONAL_MODEL_MODULES for i in selected if 0 <= i < len(MODEL_POOL)
    )
    return True


def step_tuning(state: Dict) -> Any:
    """Tuning strategy + trials + calibration + device."""
    if state.get("source") in ("demo", "full") or state.get("_from_history"):
        return SKIP

    sub = 0
    while True:
        if sub == 0:
            _clear()
            step_header(8, TOTAL_STEPS, t("s_tuning"))
            ti = select(
                [t("tune_fixed"), t("tune_random"), t("tune_optuna")],
                [t("tune_fixed_d"), t("tune_random_d"), t("tune_optuna_d")],
                title=t("pick_tuning"),
            )
            if ti < 0: return BACK
            state["hyperparam_search"] = ["fixed_grid", "random_subsample", "optuna"][ti]
            if state["hyperparam_search"] == "optuna":
                sub = 1
            else:
                sub = 2

        elif sub == 1:
            _clear()
            step_header(8, TOTAL_STEPS, t("s_tuning"))
            print(f"  {s('G', '\u2713')} {t('c_tuning')} {s('W', state['hyperparam_search'])}\n")
            print(f"  {DIM}{t('optuna_trials_hint')}{RST}")
            optuna_default = int(state.get("optuna_trials", 50))
            preset_values = [20, 50, 100, 200]
            if optuna_default not in preset_values and 1 <= optuna_default <= MAX_OPTUNA_TRIALS_INPUT:
                preset_values = [optuna_default] + preset_values
            oi = select(
                [str(v) for v in preset_values] + [t("optuna_trials_custom")],
                title=t("pick_optuna_trials_preset"),
            )
            if oi < 0:
                sub = 0
                continue
            if oi < len(preset_values):
                state["optuna_trials"] = int(preset_values[oi])
                sub = 2
                continue
            print(f"\n  {s('W', t('pick_optuna_trials'))}")
            raw = _input_line(f"  {s('C','>')} [{optuna_default}]: ")
            if raw is None:
                continue
            raw_trim = raw.strip()
            if _is_back_text_token(raw_trim):
                continue
            try:
                parsed = int(raw_trim) if raw_trim else optuna_default
            except ValueError:
                _notice(t("msg_positive_int"))
                continue
            if parsed < 1 or parsed > MAX_OPTUNA_TRIALS_INPUT:
                _notice(t("msg_optuna_trials_range"))
                continue
            state["optuna_trials"] = parsed
            sub = 2

        elif sub == 2:
            _clear()
            step_header(8, TOTAL_STEPS, t("s_tuning"))
            if strict_small_sample_active(state):
                if LANG == "zh":
                    print(f"  {s('C', '小样本严格模式已启用：将收紧模型复杂度与试验次数')}")
                else:
                    print(f"  {s('C', 'Strict small-sample mode enabled: complexity/trials will be tightened')}")
                print()
            print(f"  {s('G', '\u2713')} {t('c_tuning')} {s('W', state['hyperparam_search'])}")
            if state["hyperparam_search"] == "optuna":
                print(f"  {s('G', '\u2713')} Optuna trials: {s('W', str(state.get('optuna_trials', 50)))}")
            print()
            default_trials = int(state.get("max_trials", recommended_max_trials(state)))
            if strict_small_sample_active(state):
                default_trials = min(default_trials, STRICT_SMALL_SAMPLE_MAX_TRIALS_CAP)
            n_rows = int(state_n_rows(state) or 0)
            if n_rows > 0:
                rec_trials = recommended_max_trials(state)
                if strict_small_sample_active(state):
                    rec_trials = min(rec_trials, STRICT_SMALL_SAMPLE_MAX_TRIALS_CAP)
                rec_text = (
                    f"\u5bf9\u4e8e n={n_rows} \u7684\u63a8\u8350\u8bd5\u9a8c\u6b21\u6570: {rec_trials}"
                    if LANG == "zh"
                    else f"Recommended max trials for n={n_rows}: {rec_trials}"
                )
                print(
                    f"  {DIM}{rec_text}{RST}"
                )
            base_presets = [1, 5, 10, 20, 50]
            if strict_small_sample_active(state):
                base_presets = [v for v in base_presets if v <= STRICT_SMALL_SAMPLE_MAX_TRIALS_CAP]
                if not base_presets:
                    base_presets = [1]
            if default_trials not in base_presets and 1 <= default_trials <= MAX_TRIALS_INPUT:
                presets = [default_trials] + base_presets
            else:
                presets = list(base_presets)

            preset_labels = [str(v) for v in presets]
            choice = select(
                preset_labels + [t("trials_custom")],
                title=t("pick_trials_preset"),
            )
            if choice < 0:
                sub = 1 if state["hyperparam_search"] == "optuna" else 0
                continue
            if choice < len(presets):
                state["max_trials"] = int(presets[choice])
                sub = 3
                continue

            print(f"\n  {s('W', t('adv_trials'))}")
            raw = _input_line(f"  {s('C','>')} [{default_trials}]: ")
            if raw is None:
                continue
            raw_trim = raw.strip()
            if _is_back_text_token(raw_trim):
                continue
            try:
                parsed = int(raw_trim) if raw_trim else default_trials
            except ValueError:
                _notice(t("msg_positive_int"))
                continue
            if parsed < 1 or parsed > MAX_TRIALS_INPUT:
                _notice(t("msg_trials_range"))
                continue
            state["max_trials"] = parsed
            sub = 3

        elif sub == 3:
            _clear()
            step_header(8, TOTAL_STEPS, t("s_tuning"))
            print(f"  {s('G', '\u2713')} {t('c_tuning')} {s('W', state['hyperparam_search'])}")
            if state["hyperparam_search"] == "optuna":
                print(f"  {s('G', '\u2713')} Optuna trials: {s('W', str(state.get('optuna_trials', 50)))}")
            print(f"  {s('G', '\u2713')} {t('c_trials')} {s('W', str(state['max_trials']))}")
            print()
            ci = select(
                [t("calib_none"), t("calib_sig"), t("calib_iso"),
                 t("calib_power"), t("calib_beta")],
                [t("calib_none_d"), t("calib_sig_d"), t("calib_iso_d"),
                 t("calib_power_d"), t("calib_beta_d")],
                title=t("pick_calib"),
            )
            if ci < 0:
                sub = 2; continue
            state["calibration"] = ["none", "sigmoid", "isotonic", "power", "beta"][ci]
            sub = 4

        elif sub == 4:
            _clear()
            step_header(8, TOTAL_STEPS, t("s_tuning"))
            print(f"  {s('G', '\u2713')} {t('c_tuning')} {s('W', state['hyperparam_search'])}")
            if state["hyperparam_search"] == "optuna":
                print(f"  {s('G', '\u2713')} Optuna trials: {s('W', str(state.get('optuna_trials', 50)))}")
            print(f"  {s('G', '\u2713')} {t('c_trials')} {s('W', str(state['max_trials']))}")
            print(f"  {s('G', '\u2713')} {t('c_calib')} {s('W', state['calibration'])}")
            print()
            di = select(
                [t("dev_auto"), t("dev_cpu"), t("dev_gpu")],
                [t("dev_auto_d"), "", ""],
                title=t("pick_device"),
            )
            if di < 0:
                sub = 3; continue
            state["device"] = ["auto", "cpu", "gpu"][di]
            break
    return True


def step_advanced(state: Dict) -> Any:
    """Advanced settings — ignore-cols, n-jobs, optional backends."""
    if state.get("source") in ("demo", "full") or state.get("_from_history"):
        return SKIP

    _clear()
    step_header(9, TOTAL_STEPS, t("s_advanced"))

    pool_tokens = model_pool_tokens_from_state(state)
    has_optional_in_pool = any(family in OPTIONAL_MODEL_MODULES for family in pool_tokens)

    ci = select([t("adv_no"), t("adv_yes")], title=t("adv_ask"))
    if ci < 0:
        return BACK
    if ci == 0:
        ignore_parts = [state.get("pid", "patient_id")]
        if state.get("time"):
            ignore_parts.append(state["time"])
        state.setdefault("ignore_cols", normalize_ignore_columns(state, ignore_parts))
        state.setdefault("n_jobs", 1)
        state.setdefault("include_optional_models", has_optional_in_pool)
        enforce_optional_backend_policy(state)
        return True

    ignore_parts = [state.get("pid", "patient_id")]
    if state.get("time"):
        ignore_parts.append(state["time"])
    state.setdefault("ignore_cols", normalize_ignore_columns(state, ignore_parts))
    state.setdefault("n_jobs", 1)
    state.setdefault("include_optional_models", has_optional_in_pool)

    while True:
        _clear()
        step_header(9, TOTAL_STEPS, t("s_advanced"))
        print(f"  {s('W', t('adv_current'), bold=True)}")
        print(f"  {s('G', '\u2713')} ignore_cols = {s('W', str(state.get('ignore_cols', '')))}")
        print(f"  {s('G', '\u2713')} n_jobs = {s('W', str(state.get('n_jobs', 1)))}")
        print(f"  {s('G', '\u2713')} include_optional_models = {s('W', str(bool(state.get('include_optional_models', False))))}\n")
        ai = select(
            [t("adv_edit_ignore"), t("adv_edit_njobs"), t("adv_edit_optional"), t("adv_done")],
            title=t("adv_menu_title"),
        )
        if ai < 0:
            return BACK
        if ai == 0:
            default_ignore = str(state.get("ignore_cols", ",".join(ignore_parts)))
            mode = select(
                [t("adv_ignore_mode_select"), t("adv_ignore_mode_manual")],
                title=t("adv_ignore_mode_title"),
            )
            if mode < 0:
                continue
            if mode == 0:
                cols = available_columns_for_ignore(state)
                if not cols:
                    state["ignore_cols"] = normalize_ignore_columns(state, ignore_parts)
                    _notice(t("adv_ignore_default_applied"))
                    continue
                current_tokens = {
                    tok.strip()
                    for tok in default_ignore.split(",")
                    if tok.strip()
                }
                defaults = [idx for idx, name in enumerate(cols) if name in current_tokens]
                selected_cols = multi_select(
                    cols,
                    title=t("adv_ignore_pick_columns"),
                    defaults=defaults,
                )
                if selected_cols is None:
                    continue
                chosen = [cols[idx] for idx in selected_cols if 0 <= idx < len(cols)]
                state["ignore_cols"] = normalize_ignore_columns(state, chosen)
                continue

            print(f"\n  {s('W', t('adv_ignore'))}")
            print(f"  {DIM}{t('adv_ignore_manual_hint')}{RST}")
            raw = _input_line(f"  {s('C','>')} [{default_ignore}]: ")
            if raw is None:
                continue
            raw_trim = raw.strip()
            if _is_back_text_token(raw_trim):
                continue
            if raw_trim:
                state["ignore_cols"] = normalize_ignore_columns(state, raw_trim.split(","))
            else:
                state["ignore_cols"] = normalize_ignore_columns(state, default_ignore.split(","))
            continue
        if ai == 1:
            nj_idx = select(
                [t("adv_njobs_auto"), t("adv_njobs_1"), t("adv_njobs_4"), t("adv_njobs_8"), t("adv_njobs_custom")],
                title=t("adv_njobs"),
            )
            if nj_idx < 0:
                continue
            if nj_idx == 0:
                state["n_jobs"] = -1
                continue
            if nj_idx == 1:
                state["n_jobs"] = 1
                continue
            if nj_idx == 2:
                state["n_jobs"] = 4
                continue
            if nj_idx == 3:
                state["n_jobs"] = 8
                continue
            raw = _input_line(f"  {s('C','>')} [{state.get('n_jobs', 1)}]: ")
            if raw is None:
                continue
            raw_trim = raw.strip()
            if _is_back_text_token(raw_trim):
                continue
            try:
                parsed = int(raw_trim) if raw_trim else int(state.get("n_jobs", 1))
            except ValueError:
                _notice(t("msg_njobs_int"))
                continue
            if parsed != -1 and parsed < 1:
                _notice(t("msg_njobs_int"))
                continue
            if parsed > MAX_NJOBS_INPUT:
                _notice(t("msg_njobs_range"))
                continue
            state["n_jobs"] = parsed
            continue
        if ai == 2:
            oi = select(
                [t("adv_optional_enable"), t("adv_optional_disable")],
                [t("adv_optional_enable_d"), t("adv_optional_disable_d")],
                title=t("adv_optional"),
            )
            if oi < 0:
                continue
            if oi == 0:
                state["include_optional_models"] = True
            else:
                state["include_optional_models"] = False
                policy_result = enforce_optional_backend_policy(state)
                removed = [str(x) for x in policy_result.get("removed", []) if str(x).strip()]
                if removed:
                    print(f"\n  {s('Y', t('adv_optional_removed_notice'))} {', '.join(removed)}")
            continue
        if ai == 3:
            enforce_optional_backend_policy(state)
            break

    return True


def _export_cli(state: Dict) -> str:
    """Build a copy-ready CLI command string from wizard state."""
    import shlex as _shlex
    parts: List[str] = [sys.executable, str(SCRIPTS_DIR / "mlgg.py")]
    source = state.get("source", "")
    if source == "demo":
        parts.extend(["onboarding", "--project-root", state["out_dir"],
                       "--mode", "guided", "--yes"])
        return _shlex.join(parts)
    out_data = str(Path(state["out_dir"]) / "data")
    evidence_dir = str(Path(state["out_dir"]) / "evidence")
    models_dir = str(Path(state["out_dir"]) / "models")
    split_parts = [
        sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
        "--input", state.get("csv_path", ""),
        "--output-dir", out_data,
        "--patient-id-col", state.get("pid", "patient_id"),
        "--target-col", state.get("target", "y"),
        "--strategy", state.get("strategy", "grouped_temporal"),
        "--train-ratio", str(state.get("train_ratio", 0.6)),
        "--valid-ratio", str(state.get("valid_ratio", 0.2)),
        "--test-ratio", str(state.get("test_ratio", 0.2)),
    ]
    if state.get("time"):
        split_parts.extend(["--time-col", state["time"]])
    default_ignore = [state.get("pid", "patient_id")]
    if state.get("time"):
        default_ignore.append(state["time"])
    ignore_cols = state.get("ignore_cols", ",".join(default_ignore))
    cv_folds = state.get("cv_folds", 5)
    selection_data = "cv_inner" if state.get("validation_method") == "cv" else "valid"
    max_trials = state.get("max_trials", 20)
    imbalance_strategies = state.get("imbalance_strategies")
    if not isinstance(imbalance_strategies, list) or not imbalance_strategies:
        imbalance_strategies = [str(state.get("imbalance_strategy", "auto"))]
    imbalance_metric = str(state.get("imbalance_selection_metric", "pr_auc")).strip().lower()
    if imbalance_metric not in {"pr_auc", "roc_auc"}:
        imbalance_metric = "pr_auc"
    train_parts = [
        sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "train", "--",
        "--train", str(Path(out_data) / "train.csv"),
        "--valid", str(Path(out_data) / "valid.csv"),
        "--test", str(Path(out_data) / "test.csv"),
        "--target-col", state.get("target", "y"),
        "--patient-id-col", state.get("pid", "patient_id"),
        "--ignore-cols", ignore_cols,
        "--model-pool", state.get("model_pool", ""),
        "--hyperparam-search", state.get("hyperparam_search", "fixed_grid"),
        "--max-trials-per-family", str(max_trials),
        "--cv-splits", str(cv_folds),
        "--selection-data", selection_data,
        "--imbalance-strategy-candidates", ",".join(str(x).strip() for x in imbalance_strategies if str(x).strip()),
        "--imbalance-selection-metric", imbalance_metric,
        "--calibration-method", state.get("calibration", "none"),
        "--device", state.get("device", "auto"),
        "--model-selection-report-out", str(Path(evidence_dir) / "model_selection_report.json"),
        "--evaluation-report-out", str(Path(evidence_dir) / "evaluation_report.json"),
        "--ci-matrix-report-out", str(Path(evidence_dir) / "ci_matrix_report.json"),
        "--model-out", str(Path(models_dir) / "model.pkl"),
        "--n-jobs", str(state.get("n_jobs", 1)),
        "--random-seed", "20260225",
    ]
    if bool(state.get("include_optional_models", False)):
        train_parts.append("--include-optional-models")
    if state.get("hyperparam_search") == "optuna":
        train_parts.extend(["--optuna-trials", str(state.get("optuna_trials", 50))])
    return _shlex.join(split_parts) + "\n" + _shlex.join(train_parts)


def step_confirm(state: Dict) -> Any:
    normalize_optional_backend_state(state)
    _clear()
    title = t("s_confirm")
    if state.get("_dry_run"):
        title += f"  {s('Y', '[DRY RUN]', bold=True)}"
    step_header(10, TOTAL_STEPS, title)

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

        valid_method = state.get('validation_method', 'holdout')
        if valid_method == 'cv':
            valid_str = f"CV {state.get('cv_folds', 5)}-fold"
        else:
            valid_str = t('valid_holdout')
        imb_tokens = state.get("imbalance_strategies")
        if not isinstance(imb_tokens, list) or not imb_tokens:
            imb_tokens = [str(state.get("imbalance_strategy", "auto"))]
        imb_metric = str(state.get("imbalance_selection_metric", "pr_auc")).strip().lower()
        imb_str = ",".join(str(x) for x in imb_tokens)
        if len(imb_tokens) > 1:
            imb_str = f"{imb_str} (select_by={imb_metric})"
        trials_str = str(state.get('max_trials', 20))
        if state.get('hyperparam_search') == 'optuna':
            trials_str += f" (optuna={state.get('optuna_trials', 50)})"

        all_labels = [t('c_file'), t('c_pid'), t('c_target'), t('c_time'),
                      t('c_strat'), t('c_ratio'), t('c_validation'),
                      t('c_imbalance'), t('c_models'), t('c_tuning'),
                      t('c_trials'), t('c_calib'), t('c_device'), t('c_output')]
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
            f"{_p(t('c_validation'))}{valid_str}",
        ]
        if strict_small_sample_active(state):
            mode_label = "小样本严格模式" if LANG == "zh" else "Strict small-sample mode"
            lines.append(f"{_p(mode_label)}ON")
        lines.extend([
            "",
            f"{_p(t('c_imbalance'))}{imb_str}",
            f"{_p(t('c_models'))}{models_str}",
            f"{_p(t('c_tuning'))}{state.get('hyperparam_search', '?')}",
            f"{_p(t('c_trials'))}{trials_str}",
            f"{_p(t('c_calib'))}{state.get('calibration', '?')}",
            f"{_p(t('c_device'))}{state.get('device', '?')}",
            "",
            f"{_p(t('c_output'))}{state['out_dir']}/",
        ])
        box(t("s_confirm"), lines, color="C")

    print()
    while True:
        ci = select([t("c_start"), t("c_export"), t("c_back")])
        if ci == 2 or ci < 0:
            return BACK
        if ci == 1:
            print(f"\n  {s('W', _export_cli(state))}")
            sys.stdout.write(SHOW_CUR)
            try:
                input(f"  {DIM}{t('enter_continue')}{RST}")
            except (EOFError, KeyboardInterrupt):
                pass
            _clear()
            step_header(10, TOTAL_STEPS, t("s_confirm"))
            continue
        break
    return True


_ERROR_PATTERNS = [
    ("network error", {
        "en": "Dataset download failed due to network/DNS. In play mode, use built-in stable datasets: heart, breast, ckd.",
        "zh": "\u6570\u636e\u96c6\u4e0b\u8f7d\u5931\u8d25\uff08\u7f51\u7edc/DNS \u95ee\u9898\uff09\u3002play \u6a21\u5f0f\u8bf7\u4f18\u5148\u4f7f\u7528\u7a33\u5b9a\u5185\u7f6e\u6570\u636e\u96c6\uff1aheart\u3001breast\u3001ckd\u3002",
    }),
    ("nodename nor servname", {
        "en": "DNS lookup failed. Check network, or choose built-in stable datasets (heart/breast/ckd).",
        "zh": "DNS \u89e3\u6790\u5931\u8d25\u3002\u8bf7\u68c0\u67e5\u7f51\u7edc\uff0c\u6216\u6539\u7528\u7a33\u5b9a\u5185\u7f6e\u6570\u636e\u96c6\uff08heart/breast/ckd\uff09\u3002",
    }),
    ("not enough positive", {
        "en": "Not enough positive (y=1) samples in one or more splits. Try a larger dataset or adjust ratios.",
        "zh": "一个或多个分割中正样本（y=1）不足。请尝试更大的数据集或调整比例。",
    }),
    ("not enough negative", {
        "en": "Not enough negative (y=0) samples in one or more splits. Check class balance.",
        "zh": "一个或多个分割中负样本（y=0）不足。请检查类别平衡。",
    }),
    ("FileNotFoundError", {
        "en": "A required file was not found. Check input paths and previous steps.",
        "zh": "未找到必需的文件。请检查输入路径和前面的步骤。",
    }),
    ("column", {
        "en": "A required column was not found in the CSV. Verify column names match your data.",
        "zh": "CSV 中未找到必需的列。请确认列名与数据匹配。",
    }),
    ("PermissionError", {
        "en": "Permission denied writing output files. Check directory permissions.",
        "zh": "写入输出文件时权限被拒绝。请检查目录权限。",
    }),
    ("ModuleNotFoundError", {
        "en": "A required Python package is missing. Run: pip install -r requirements.txt",
        "zh": "缺少必需的 Python 包。请运行：pip install -r requirements.txt",
    }),
]


def _friendly_error(err: str) -> str:
    """Match stderr against known patterns and return a friendly hint."""
    lower = err.lower()
    for pattern, msgs in _ERROR_PATTERNS:
        if pattern.lower() in lower:
            return msgs.get(LANG, msgs["en"])
    return ""


def _override_cli_arg(cmd: List[str], flag: str, value: str) -> List[str]:
    """Return a copied command list with one --flag value updated/inserted."""
    out = list(cmd)
    for idx, token in enumerate(out):
        if token == flag and idx + 1 < len(out):
            out[idx + 1] = value
            return out
    out.extend([flag, value])
    return out


def step_run(state: Dict) -> Any:
    normalize_optional_backend_state(state)
    _clear()
    step_header(11, TOTAL_STEPS, t("s_run"))

    source = state["source"]

    # ── Dry-run: print config summary + commands, save history ──
    if state.get("_dry_run"):
        import shlex as _shlex
        # Config summary box
        dk = state.get("dataset_key", "custom")
        if dk != "demo":
            fname = Path(state.get("csv_path", "")).name or "?"
            ratio_str = f"{int(state.get('train_ratio',0.6)*100)}/{int(state.get('valid_ratio',0.2)*100)}/{int(state.get('test_ratio',0.2)*100)}"
            models_str = ", ".join(state.get("_model_labels", ["?"]))
            vm = state.get('validation_method', 'holdout')
            valid_str = f"CV {state.get('cv_folds', 5)}-fold" if vm == 'cv' else t('valid_holdout')
            imb_tokens = state.get("imbalance_strategies")
            if not isinstance(imb_tokens, list) or not imb_tokens:
                imb_tokens = [str(state.get("imbalance_strategy", "auto"))]
            imb_metric = str(state.get("imbalance_selection_metric", "pr_auc")).strip().lower()
            imb_str = ",".join(str(x) for x in imb_tokens)
            if len(imb_tokens) > 1:
                imb_str = f"{imb_str} (select_by={imb_metric})"
            trials_str = str(state.get('max_trials', 20))
            if state.get('hyperparam_search') == 'optuna':
                trials_str += f" (optuna={state.get('optuna_trials', 50)})"
            all_labels = [t('c_file'), t('c_pid'), t('c_target'), t('c_time'),
                          t('c_strat'), t('c_ratio'), t('c_validation'),
                          t('c_imbalance'), t('c_models'), t('c_tuning'),
                          t('c_trials'), t('c_calib'), t('c_device'), t('c_output')]
            col_w = max(_wlen(l) for l in all_labels) + 2
            def _p(label: str) -> str:
                return label + " " * max(col_w - _wlen(label), 1)
            summary_lines = [
                f"{_p(t('c_file'))}{fname}",
                f"{_p(t('c_pid'))}{state.get('pid', '?')}",
                f"{_p(t('c_target'))}{state.get('target', '?')}",
                f"{_p(t('c_time'))}{state.get('time') or t('c_none')}",
                f"{_p(t('c_strat'))}{state.get('strategy', '?')}",
                f"{_p(t('c_ratio'))}{ratio_str}",
                f"{_p(t('c_validation'))}{valid_str}",
                "",
                f"{_p(t('c_imbalance'))}{imb_str}",
                f"{_p(t('c_models'))}{models_str}",
                f"{_p(t('c_tuning'))}{state.get('hyperparam_search', '?')}",
                f"{_p(t('c_trials'))}{trials_str}",
                f"{_p(t('c_calib'))}{state.get('calibration', '?')}",
                f"{_p(t('c_device'))}{state.get('device', '?')}",
                "",
                f"{_p(t('c_output'))}{state['out_dir']}/",
            ]
            box(t("s_confirm"), summary_lines, color="C")
            print()

        # CLI commands
        print(f"  {s('Y', '[DRY RUN]', bold=True)} — commands NOT executed:\n")
        cli_str = _export_cli(state)
        for line in cli_str.split("\n"):
            print(f"  {s('W', line)}")
        print()
        print(f"  {s('G', t('r_dry_done'))}")

        # Save history even in dry-run
        state.pop("_from_history", None)
        _save_history(state)
        return True

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
            # Show metrics from demo evaluation report
            try:
                import json as _json
                demo_eval = Path(state["out_dir"]) / "evidence" / "evaluation_report.json"
                if demo_eval.exists():
                    ed = _json.loads(demo_eval.read_text())
                    dm = ed.get("metrics", {})
                    if isinstance(dm, dict):
                        dl = []
                        mid = ed.get("model_id")
                        if mid:
                            dl.append(f"  {'Model':<14} {mid}")
                            dl.append("")
                        for k, lb in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
                                      ("f1", "F1"), ("sensitivity", "Sensitivity"),
                                      ("specificity", "Specificity"), ("accuracy", "Accuracy")]:
                            v = dm.get(k)
                            if v is not None:
                                dl.append(f"  {lb:<14} {float(v):.4f}")
                        if dl:
                            print()
                            box(t("r_metrics"), dl, color="C")
            except Exception:
                pass
        else:
            print(f"  {s('R', t('x_fail'))}")
            if err:
                for l in err.strip().split("\n")[-5:]:
                    print(f"  {DIM}{l}{RST}")
            return FAIL
        return True

    # ── Full Publication-Grade Pipeline ──
    if source == "full":
        project_root = state.get("_full_project_root", state["out_dir"])
        request_json = state.get("_full_request_json", "")
        evidence_dir = str(Path(project_root) / "evidence")
        workflow_script = SCRIPTS_DIR / "run_productized_workflow.py"
        if not workflow_script.exists():
            workflow_script = SCRIPTS_DIR / "run_strict_pipeline.py"
        cmd = [
            sys.executable, str(workflow_script),
            "--request", request_json,
            "--evidence-dir", evidence_dir,
            "--strict",
            "--allow-missing-compare",
            "--report", str(Path(evidence_dir) / "strict_pipeline_report.json"),
        ]
        rc, _, err = run_spinner(cmd, t("x_pipeline_full"))
        print()
        if rc == 0:
            box(t("r_done"), [
                f"{t('c_output')} {project_root}/",
                "  evidence/  -- all gate reports",
            ], color="G")
        else:
            print(f"  {s('R', t('x_fail'))}")
            if err:
                for l in err.strip().split("\n")[-5:]:
                    print(f"  {DIM}{l}{RST}")
            return FAIL
        return True

    # ── Download + Split + Train flow ──
    completed: List[Tuple[str, str]] = []  # (label, "done"|"fail")
    out_data = str(Path(state["out_dir"]) / "data")
    total_phases = 3 if source == "download" else 2

    def _progress() -> None:
        done_count = sum(1 for _, st in completed if st == "done")
        pct = int(done_count * 100 / total_phases) if total_phases else 0
        bar_w = 20
        filled = int(bar_w * pct / 100)
        bar = s('G', '\u2588' * filled) + s('W', '\u2591' * (bar_w - filled))
        print(f"  {bar} {s('W', f'{pct}%')}")
        for label, st in completed:
            icon = s('G', '\u2713') if st == "done" else s('R', '\u2717')
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
            _clear(); step_header(11, TOTAL_STEPS, t("s_run")); _progress()
            print(f"\n  {s('R', t('x_fail'))}")
            hint = _friendly_error(err) if err else ""
            if hint:
                print(f"  {s('C', hint)}")
            if err:
                print()
                for l in err.strip().split("\n")[-3:]:
                    print(f"  {DIM}{l}{RST}")
            return FAIL
        completed.append((dl_label, "done"))

    # Apply strict small-sample profile before dependency checks so
    # optional backends are not requested for models that will be pruned anyway.
    strict_profile = apply_strict_small_sample_profile(state)
    if strict_profile.get("active"):
        # Keep labels consistent with any enforced model-pool adjustment.
        pool_now = [token.strip() for token in str(state.get("model_pool", "")).split(",") if token.strip()]
        label_map = {name: t(label_key) for name, label_key in MODEL_POOL}
        state["_model_labels"] = [label_map.get(name, name) for name in pool_now]

    # Resolve runtime dependencies before split/train to fail early.
    if not ensure_runtime_dependencies(state):
        print(f"  {s('R', t('dep_cancelled'))}")
        return FAIL

    backend_prune = prune_unavailable_optional_models(state)
    if backend_prune.get("removed"):
        removed_text = ", ".join(str(x) for x in backend_prune["removed"])
        kept_text = ", ".join(str(x) for x in backend_prune["kept"])
        if LANG == "zh":
            print(f"  {s('Y', '提示：检测到未安装的可选模型后端，已自动移除：')} {removed_text}")
            print(f"  {s('C', '继续训练模型池：')} {kept_text}")
            if backend_prune.get("fallback_used"):
                print(f"  {s('C', '未保留可用模型，已自动回退到 logistic_l2。')}")
        else:
            print(f"  {s('Y', 'Notice: unavailable optional model backends were removed:')} {removed_text}")
            print(f"  {s('C', 'Continuing with model pool:')} {kept_text}")
            if backend_prune.get("fallback_used"):
                print(f"  {s('C', 'No usable model remained; auto-fallback to logistic_l2.')}")
        print()

    # ── Phase 2: Split ──
    _clear(); step_header(11, TOTAL_STEPS, t("s_run")); _progress()
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
        _clear(); step_header(11, TOTAL_STEPS, t("s_run")); _progress()
        print(f"\n  {s('R', t('x_fail'))}")
        hint = _friendly_error(err) if err else ""
        if hint:
            print(f"  {s('C', hint)}")
        if err:
            print()
            for l in err.strip().split("\n")[-3:]:
                print(f"  {DIM}{l}{RST}")
        return FAIL
    completed.append((split_label, "done"))

    # Show split results
    _clear(); step_header(11, TOTAL_STEPS, t("s_run")); _progress()
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
    model_count = len([x for x in str(state.get("model_pool", "")).split(",") if x.strip()])
    train_label = t("x_train", n=model_count)

    evidence_dir = str(Path(state["out_dir"]) / "evidence")
    models_dir = str(Path(state["out_dir"]) / "models")
    Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Use user-customized ignore_cols from step_advanced, or build default
    default_ignore_parts = [state["pid"]]
    if state.get("time"):
        default_ignore_parts.append(state["time"])
    if source == "download" and "event_time" not in default_ignore_parts:
        default_ignore_parts.append("event_time")
    ignore_cols = state.get("ignore_cols", ",".join(default_ignore_parts))

    cv_folds = state.get("cv_folds", 5)
    selection_data = "cv_inner" if state.get("validation_method") == "cv" else "valid"
    max_trials = state.get("max_trials", 20)
    imbalance_strategies = state.get("imbalance_strategies")
    if not isinstance(imbalance_strategies, list) or not imbalance_strategies:
        imbalance_strategies = [str(state.get("imbalance_strategy", "auto"))]
    imbalance_metric = str(state.get("imbalance_selection_metric", "pr_auc")).strip().lower()
    if imbalance_metric not in {"pr_auc", "roc_auc"}:
        imbalance_metric = "pr_auc"

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
        "--max-trials-per-family", str(max_trials),
        "--cv-splits", str(cv_folds),
        "--selection-data", selection_data,
        "--imbalance-strategy-candidates", ",".join(str(x).strip() for x in imbalance_strategies if str(x).strip()),
        "--imbalance-selection-metric", imbalance_metric,
        "--calibration-method", state["calibration"],
        "--device", state["device"],
        "--model-selection-report-out", str(Path(evidence_dir) / "model_selection_report.json"),
        "--evaluation-report-out", str(Path(evidence_dir) / "evaluation_report.json"),
        "--ci-matrix-report-out", str(Path(evidence_dir) / "ci_matrix_report.json"),
        "--model-out", str(Path(models_dir) / "model.pkl"),
        "--n-jobs", str(state.get("n_jobs", 1)),
        "--random-seed", "20260225",
    ]
    if bool(state.get("include_optional_models", False)):
        train_cmd.append("--include-optional-models")
    if state.get("hyperparam_search") == "optuna":
        train_cmd.extend(["--optuna-trials", str(state.get("optuna_trials", 50))])

    rc, _, err = run_with_progress(train_cmd, train_label, total=model_count)
    if rc != 0:
        completed.append((train_label, "fail"))
        _clear(); step_header(11, TOTAL_STEPS, t("s_run")); _progress()
        print(f"\n  {s('R', t('x_fail'))}")
        hint = _friendly_error(err) if err else ""
        if hint:
            print(f"  {s('C', hint)}")
        if err:
            print()
            for l in err.strip().split("\n")[-5:]:
                print(f"  {DIM}{l}{RST}")
        return FAIL
    completed.append((train_label, "done"))

    # Show final results
    _clear(); step_header(11, TOTAL_STEPS, t("s_run")); _progress()
    if strict_profile.get("active"):
        applied_items = strict_profile.get("applied", [])
        if isinstance(applied_items, list) and applied_items:
            if LANG == "zh":
                print(
                    f"\n  {s('C', '小样本严格模式已生效')}: "
                    f"n={strict_profile.get('n_rows', '?')}, "
                    f"阈值<={strict_profile.get('max_rows', '?')}"
                )
            else:
                print(
                    f"\n  {s('C', 'Strict small-sample profile applied')}: "
                    f"n={strict_profile.get('n_rows', '?')}, "
                    f"threshold<={strict_profile.get('max_rows', '?')}"
                )
    print()
    box(t("r_train_ok"), [
        f"{t('c_output')} {state['out_dir']}/",
        "  evidence/  -- model_selection_report, evaluation_report",
        "  models/    -- trained model artifact",
        "  data/      -- train / valid / test splits",
    ], color="G")

    play_blockers: List[str] = []
    play_advisories: List[str] = []
    readiness_evaluated = False
    readiness_error = ""

    # Show key metrics from evaluation report
    try:
        eval_path = Path(evidence_dir) / "evaluation_report.json"
        if eval_path.exists():
            import json as _json
            eval_data = _json.loads(eval_path.read_text())
            metrics = eval_data.get("metrics", {})
            has_core_metric = isinstance(metrics, dict) and any(
                metrics.get(k) is not None for k in ("pr_auc", "roc_auc", "f1", "brier")
            )
            if not has_core_metric:
                readiness_error = "evaluation_report_schema_invalid"
            else:
                readiness_evaluated = True
                metric_lines = []
                # Model ID
                mid = eval_data.get("model_id")
                if mid:
                    metric_lines.append(f"  {'Model':<14} {mid}")
                # Threshold
                thresh_block = eval_data.get("threshold_selection", {})
                thresh_val = thresh_block.get("selected_threshold")
                if thresh_val is not None:
                    metric_lines.append(f"  {'Threshold':<14} {float(thresh_val):.4f}")
                if mid or thresh_val is not None:
                    metric_lines.append("")
                # Core metrics with 95% CI
                unc_metrics = eval_data.get("uncertainty", {}).get("metrics", {})
                for key, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
                                   ("f1", "F1"), ("f2_beta", "F-beta"),
                                   ("accuracy", "Accuracy"),
                                   ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                                   ("ppv", "PPV"), ("npv", "NPV"),
                                   ("brier", "Brier")]:
                    val = metrics.get(key)
                    if val is not None:
                        line = f"  {label:<14} {float(val):.4f}"
                        ci = unc_metrics.get(key, {}).get("ci_95")
                        if isinstance(ci, list) and len(ci) == 2 and ci[0] is not None:
                            line += f"  [{float(ci[0]):.4f}-{float(ci[1]):.4f}]"
                        metric_lines.append(line)
                # CI method note
                unc_method = eval_data.get("uncertainty", {}).get("method", "")
                unc_n = eval_data.get("uncertainty", {}).get("n_resamples", 0)
                if unc_method == "bootstrap" and unc_n:
                    metric_lines.append("")
                    metric_lines.append(f"  {DIM}95% CI: bootstrap, n={unc_n}{RST}")
                # Constraints status
                constraints_ok = thresh_block.get("constraints_satisfied_overall")
                if constraints_ok is not None:
                    status = s('G', 'PASS') if constraints_ok else s('R', 'FAIL')
                    metric_lines.append("")
                    metric_lines.append(f"  {'Constraints':<14} {status}")
                if metric_lines:
                    print()
                    box(t("r_metrics"), metric_lines, color="C")

                # Per-split comparison with Gap (overfitting check)
                split_metrics = eval_data.get("split_metrics", {})
                train_m = split_metrics.get("train", {}).get("metrics", {})
                valid_m = split_metrics.get("valid", {}).get("metrics", {})
                test_m = split_metrics.get("test", {}).get("metrics", {})
                overfit = eval_data.get("overfitting_analysis", {})
                gaps = overfit.get("gaps", {})
                if train_m and test_m:
                    cmp_lines = []
                    cmp_lines.append(f"  {'':14} {'Train':>8}  {'Valid':>8}  {'Test':>8}  {'Gap':>8}")
                    for key, label in [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"),
                                       ("f1", "F1"), ("brier", "Brier")]:
                        tv = train_m.get(key)
                        vv = valid_m.get(key)
                        ev = test_m.get(key)
                        if tv is not None and ev is not None:
                            vv_str = f"{float(vv):.4f}" if vv is not None else "  --  "
                            # Gap = train - test (for Brier: lower is better, so negative gap = overfitting)
                            gap_val = gaps.get(key, {}).get("train_test_gap")
                            if gap_val is not None:
                                gap_f = float(gap_val)
                                # Flag: for AUC/F1 gap>0.10 is concerning; for Brier gap<-0.05
                                if key == "brier":
                                    flag = s('Y', f"{gap_f:>+8.4f}") if gap_f < -0.05 else f"{gap_f:>+8.4f}"
                                else:
                                    flag = s('Y', f"{gap_f:>+8.4f}") if gap_f > 0.10 else f"{gap_f:>+8.4f}"
                            else:
                                gap_f = float(tv) - float(ev)
                                flag = f"{gap_f:>+8.4f}"
                            cmp_lines.append(f"  {label:<14} {float(tv):>8.4f}  {vv_str:>8}  {float(ev):>8.4f}  {flag}")
                    # Risk level + warnings + recommendations
                    risk = overfit.get("risk_level", "low")
                    warnings = overfit.get("warnings", [])
                    recs = overfit.get("recommendations", [])
                    cmp_lines.append("")
                    if risk == "high":
                        cmp_lines.append(f"  {s('R', f'Risk: {risk.upper()}', bold=True)}")
                    elif risk == "medium":
                        cmp_lines.append(f"  {s('Y', f'Risk: {risk.upper()}', bold=True)}")
                    else:
                        cmp_lines.append(f"  {s('G', 'Risk: LOW — No overfitting detected')}")
                    for w in warnings:
                        cmp_lines.append(f"  {s('Y', w)}")
                    if recs:
                        cmp_lines.append("")
                        for r in recs:
                            cmp_lines.append(f"  {DIM}> {r}{RST}")
                    if len(cmp_lines) > 1:
                        print()
                        box("Train / Valid / Test", cmp_lines, color="C")

                # Fallback trace (overfitting callback)
                if overfit.get("callback_activated") and overfit.get("fallback_trace"):
                    ft = overfit["fallback_trace"]
                    orig = overfit.get("original_model_id", "?")
                    fb_lines = []
                    fb_lines.append(f"  {'Original':<14} {orig}")
                    fb_lines.append(f"  {'Final':<14} {eval_data.get('model_id', '?')}")
                    switched = orig != eval_data.get("model_id")
                    if switched:
                        fb_lines.append(f"  {'Status':<14} {s('G', 'Switched to less overfitting model')}")
                    else:
                        fb_lines.append(f"  {'Status':<14} {s('Y', 'No better alternative found')}")
                    fb_lines.append("")
                    fb_lines.append(f"  {'#':>3}  {'Model':<22} {'Risk':>6}  {'Gap':>8}  {'PR-AUC':>8}")
                    for step in ft:
                        r = step.get("round", "?")
                        mid = str(step.get("model_id", "?"))[:22]
                        sr = str(step.get("risk", "?"))
                        mg = step.get("max_gap")
                        tp = step.get("test_pr_auc")
                        mg_s = f"{float(mg):>+8.4f}" if mg is not None else "     --"
                        tp_s = f"{float(tp):>8.4f}" if tp is not None else "     --"
                        if sr == "low":
                            sr_s = s('G', f"{sr:>6}")
                        elif sr == "high":
                            sr_s = s('R', f"{sr:>6}")
                        elif sr == "medium":
                            sr_s = s('Y', f"{sr:>6}")
                        else:
                            sr_s = f"{sr:>6}"
                        fb_lines.append(f"  {str(r):>3}  {mid:<22} {sr_s}  {mg_s}  {tp_s}")
                    print()
                    box("Overfitting Callback", fb_lines, color="C")

                # TRIPOD+AI supplementary assessments
                cal = eval_data.get("calibration_assessment", {})
                epv = eval_data.get("sample_size_adequacy", {})
                vif = eval_data.get("multicollinearity", {})
                tripod_warnings: List[str] = []
                tripod_blockers: List[str] = []
                if cal or epv:
                    ta_lines = []
                    # Calibration
                    slope = cal.get("calibration_slope")
                    intercept = cal.get("calibration_intercept")
                    eo = cal.get("expected_observed_ratio")
                    if slope is not None:
                        slope_v = float(slope)
                        slope_ok = 0.8 <= slope_v <= 1.2
                        slope_bad = slope_v < 0.5 or slope_v > 1.5
                        if slope_ok:
                            sl = s('G', f"{slope_v:.4f}")
                            tag = s('G', "OK")
                        elif slope_bad:
                            sl = s('R', f"{slope_v:.4f}")
                            tag = s('R', "FAIL")
                            tripod_blockers.append("calibration_slope")
                        else:
                            sl = s('Y', f"{slope_v:.4f}")
                            tag = s('Y', "WARN")
                            tripod_warnings.append("calibration_slope")
                        ta_lines.append(f"  {'Cal. slope':<18} {sl}  {'(ideal=1.0)':>14}  [{tag}]")
                    if intercept is not None:
                        int_v = float(intercept)
                        int_ok = abs(int_v) < 0.1
                        int_bad = abs(int_v) > 1.0
                        if int_ok:
                            il = s('G', f"{int_v:.4f}")
                            tag = s('G', "OK")
                        elif int_bad:
                            il = s('R', f"{int_v:.4f}")
                            tag = s('R', "FAIL")
                            tripod_blockers.append("calibration_intercept")
                        else:
                            il = s('Y', f"{int_v:.4f}")
                            tag = s('Y', "WARN")
                            tripod_warnings.append("calibration_intercept")
                        ta_lines.append(f"  {'Cal. intercept':<18} {il}  {'(ideal=0.0)':>14}  [{tag}]")
                    if eo is not None:
                        eo_v = float(eo)
                        eo_ok = 0.8 <= eo_v <= 1.2
                        el = s('G', f"{eo_v:.4f}") if eo_ok else s('Y', f"{eo_v:.4f}")
                        tag = s('G', "OK") if eo_ok else s('Y', "WARN")
                        if not eo_ok:
                            tripod_warnings.append("eo_ratio")
                        ta_lines.append(f"  {'E:O ratio':<18} {el}  {'(ideal=1.0)':>14}  [{tag}]")
                    ece_val = cal.get("ece")
                    if ece_val is not None:
                        ece_f = float(ece_val)
                        ece_ok = ece_f <= 0.05
                        ece_bad = ece_f > 0.10
                        if ece_ok:
                            ece_s = s('G', f"{ece_f:.4f}")
                            tag = s('G', "OK")
                        elif ece_bad:
                            ece_s = s('R', f"{ece_f:.4f}")
                            tag = s('R', "FAIL")
                            tripod_blockers.append("ece")
                        else:
                            ece_s = s('Y', f"{ece_f:.4f}")
                            tag = s('Y', "WARN")
                            tripod_warnings.append("ece")
                        ta_lines.append(f"  {'ECE':<18} {ece_s}  {'(ideal<0.05)':>14}  [{tag}]")
                    # EPV
                    if epv.get("events_per_variable") is not None:
                        epv_v = float(epv["events_per_variable"])
                        adq = str(epv.get("adequacy", "?"))
                        if adq == "adequate":
                            epv_s = s('G', f"{epv_v:.1f} ({adq})")
                            tag = s('G', "OK")
                        elif adq == "marginal":
                            epv_s = s('Y', f"{epv_v:.1f} ({adq})")
                            tag = s('Y', "WARN")
                            tripod_warnings.append("epv")
                        else:
                            epv_s = s('R', f"{epv_v:.1f} ({adq})")
                            tag = s('R', "FAIL")
                            tripod_blockers.append("epv")
                        ta_lines.append(f"  {'EPV':<18} {epv_s}  [{tag}]")
                    # VIF
                    if not vif.get("skipped", True):
                        max_vif = vif.get("max_vif", 0)
                        hvc = vif.get("high_vif_count", 0)
                        if hvc > 0:
                            if int(hvc) >= 10:
                                tag = s('R', "FAIL")
                                tripod_blockers.append("vif")
                            else:
                                tag = s('Y', "WARN")
                                tripod_warnings.append("vif")
                            ta_lines.append(
                                f"  {'VIF max':<18} {s('Y', f'{float(max_vif):.1f}')}  {s('Y', f'{hvc} features >10')}  [{tag}]"
                            )
                        else:
                            ta_lines.append(f"  {'VIF max':<18} {s('G', f'{float(max_vif):.1f}')}  [{s('G', 'OK')}]")
                    # NRI
                    nri_data = eval_data.get("net_reclassification_improvement", {})
                    nri_log = nri_data.get("vs_logistic_baseline", {})
                    if nri_log.get("nri_total") is not None:
                        nv = float(nri_log["nri_total"])
                        cv = nri_log.get("continuous_nri_total")
                        nri_col = 'G' if nv > 0 else ('Y' if nv == 0 else 'R')
                        nri_s = s(nri_col, f"{nv:+.4f}")
                        ta_lines.append(f"  {'NRI vs logistic':<18} {nri_s}")
                        if cv is not None:
                            cv_col = 'G' if float(cv) > 0 else 'Y'
                            ta_lines.append(f"  {'  continuous':<18} {s(cv_col, f'{float(cv):+.4f}')}")
                    # DCA net benefit summary at selected threshold
                    dca = eval_data.get("decision_curve_analysis", {})
                    dca_pts = dca.get("thresholds", [])
                    sel_thresh = eval_data.get("threshold_selection", {}).get("selected_threshold")
                    if dca_pts and sel_thresh is not None:
                        closest = min(dca_pts, key=lambda r: abs(r["threshold"] - float(sel_thresh)))
                        nb = closest.get("net_benefit_model")
                        nb_all = closest.get("net_benefit_treat_all")
                        if nb is not None:
                            nb_col = 'G' if float(nb) > max(float(nb_all or 0), 0) else 'Y'
                            ta_lines.append(f"  {'DCA net benefit':<18} {s(nb_col, f'{float(nb):.4f}')}"
                                            f"  {'(at threshold)':>14}")
                    # Permutation importance top-3
                    perm = eval_data.get("permutation_importance", {})
                    top_feats = perm.get("top_features", [])
                    if top_feats:
                        top3 = top_feats[:3]
                        for idx, f in enumerate(top3, start=1):
                            f_name = str(f.get("feature", "?"))[:16]
                            f_imp = float(f.get("importance_mean", 0.0))
                            key = "Perm. imp top1" if idx == 1 else ("top2" if idx == 2 else "top3")
                            ta_lines.append(f"  {key:<18} {f_name}={f_imp:.3f}")
                    if ta_lines:
                        print()
                        box("TRIPOD+AI Checks", ta_lines, color="C")

                # Statistical tests + top-conference metrics
                st_lines = []
                stat_tests = eval_data.get("statistical_tests", {})
                dl = stat_tests.get("delong_vs_logistic", {})
                mn = stat_tests.get("mcnemar_vs_logistic", {})
                if dl.get("p_value") is not None:
                    dl_p = float(dl["p_value"])
                    dl_sig = dl.get("significant_at_005", False)
                    dl_col = 'G' if dl_sig else 'Y'
                    sig_s = 'sig' if dl_sig else 'n.s.'
                    st_lines.append(f"  {'DeLong p-value':<20} {s(dl_col, f'{dl_p:.4f}')}  {s(dl_col, sig_s)}")
                if mn.get("p_value") is not None:
                    mn_p = float(mn["p_value"])
                    mn_sig = mn.get("significant_at_005", False)
                    mn_col = 'G' if mn_sig else 'Y'
                    sig_s = 'sig' if mn_sig else 'n.s.'
                    st_lines.append(f"  {'McNemar p-value':<20} {s(mn_col, f'{mn_p:.4f}')}  {s(mn_col, sig_s)}")
                pu = eval_data.get("prediction_uncertainty", {})
                if pu.get("entropy_mean") is not None:
                    hu = pu.get("high_uncertainty_fraction", 0)
                    hu_col = 'G' if float(hu) < 0.1 else ('Y' if float(hu) < 0.3 else 'R')
                    st_lines.append(f"  {'Entropy (mean)':<20} {pu['entropy_mean']:.4f}"
                                    f"  high-unc={s(hu_col, f'{float(hu):.1%}')}")
                sg = eval_data.get("subgroup_performance", {})
                n_sg = sg.get("features_analyzed", 0)
                if n_sg > 0:
                    dr = sg.get("disparate_impact_ratio")
                    if dr is not None:
                        dr_col = 'G' if float(dr) >= 0.8 else ('Y' if float(dr) >= 0.6 else 'R')
                        st_lines.append(f"  {'Disparate impact':<20} {s(dr_col, f'{float(dr):.4f}')}"
                                        f"  ({n_sg} features)")
                ib = eval_data.get("inference_benchmark", {})
                lat = ib.get("inference_latency_ms_per_sample")
                if lat is not None:
                    pc = ib.get("model_param_count")
                    pc_s = f"  params={pc}" if pc is not None else ""
                    st_lines.append(f"  {'Inference latency':<20} {float(lat):.3f} ms/sample{pc_s}")
                if st_lines:
                    print()
                    box("Statistical Tests & Fairness", st_lines, color="C")

                # Release readiness summary: aggregate key gates into one compact, actionable view.
                readiness_lines = []
                blockers: List[str] = []
                advisories: List[str] = []
                if constraints_ok is False:
                    blockers.append("threshold_constraints")
                elif constraints_ok is None:
                    advisories.append("threshold_constraints_unknown")
                if str(overfit.get("risk_level", "")).lower() == "high":
                    blockers.append("overfitting_high_risk")
                elif str(overfit.get("risk_level", "")).lower() == "medium":
                    advisories.append("overfitting_medium_risk")
                blockers.extend([x for x in tripod_blockers if x not in blockers])
                advisories.extend([x for x in tripod_warnings if x not in advisories and x not in blockers])
                play_blockers = list(blockers)
                play_advisories = list(advisories)

                if blockers:
                    overall_tag = s('R', t("r_play_status_not_ready"), bold=True)
                    verdict = t("r_verdict_not_ready")
                elif advisories:
                    overall_tag = s('Y', t("r_play_status_warn"), bold=True)
                    verdict = t("r_verdict_warn")
                else:
                    overall_tag = s('G', t("r_play_status_pass"), bold=True)
                    verdict = t("r_verdict_pass")
                readiness_lines.append(f"  {'Overall':<16} {overall_tag}  {verdict}")
                readiness_lines.append(f"  {'Constraints':<16} {s('G','PASS') if constraints_ok else s('R','FAIL') if constraints_ok is False else s('Y','N/A')}")
                readiness_lines.append(f"  {'Overfitting':<16} {str(overfit.get('risk_level', 'unknown')).upper()}")
                readiness_lines.append(f"  {t('r_pub_gate_not_run_label'):<16} {s('Y', t('r_pub_gate_not_run_value'))}")
                calibration_blockers = {
                    "calibration_slope",
                    "calibration_intercept",
                    "ece",
                    "epv",
                }
                if not (set(blockers) & {"overfitting_high_risk"}) and (set(blockers) & calibration_blockers):
                    primary_issue_text = "校准/样本充足性" if LANG == "zh" else "Calibration / sample adequacy"
                    primary_issue_label = "主要问题" if LANG == "zh" else "Primary issue"
                    readiness_lines.append(
                        f"  {primary_issue_label:<16} "
                        f"{s('Y', primary_issue_text)}"
                    )
                if blockers:
                    readiness_lines.append("")
                    readiness_lines.append(f"  {s('R', 'Blocking items:')}")
                    for b in blockers[:4]:
                        readiness_lines.append(f"  - {b}")
                elif advisories:
                    readiness_lines.append("")
                    readiness_lines.append(f"  {s('Y', 'Watch items:')}")
                    for w in advisories[:4]:
                        readiness_lines.append(f"  - {w}")
                if set(blockers) & calibration_blockers:
                    suggested_profile_title = "建议复跑配置：" if LANG == "zh" else "Suggested rerun profile:"
                    suggested_models = "  - 模型：logistic_l2, logistic_elasticnet" if LANG == "zh" else "  - models: logistic_l2, logistic_elasticnet"
                    suggested_trials = "  - max_trials_per_family：<= 8" if LANG == "zh" else "  - max_trials_per_family: <= 8"
                    suggested_calib = "  - 校准：none 或 power" if LANG == "zh" else "  - calibration: none or power"
                    readiness_lines.append("")
                    readiness_lines.append(f"  {s('C', suggested_profile_title)}")
                    readiness_lines.append(suggested_models)
                    readiness_lines.append(suggested_trials)
                    readiness_lines.append(suggested_calib)
                    # Emit copy-ready rerun commands (same split files, conservative profile).
                    import shlex as _shlex
                    conservative_trials = min(
                        STRICT_SMALL_SAMPLE_MAX_TRIALS_CAP,
                        max(1, int(recommended_max_trials(state))),
                    )
                    model_pool_conservative = "logistic_l2,logistic_elasticnet"
                    rerun_power = list(train_cmd)
                    rerun_power = _override_cli_arg(rerun_power, "--model-pool", model_pool_conservative)
                    rerun_power = _override_cli_arg(rerun_power, "--hyperparam-search", "random_subsample")
                    rerun_power = _override_cli_arg(rerun_power, "--max-trials-per-family", str(conservative_trials))
                    rerun_power = _override_cli_arg(rerun_power, "--calibration-method", "power")
                    rerun_none = list(train_cmd)
                    rerun_none = _override_cli_arg(rerun_none, "--model-pool", "logistic_l2")
                    rerun_none = _override_cli_arg(rerun_none, "--hyperparam-search", "fixed_grid")
                    rerun_none = _override_cli_arg(rerun_none, "--max-trials-per-family", "1")
                    rerun_none = _override_cli_arg(rerun_none, "--calibration-method", "none")
                    rerun_cmd_power = _shlex.join(rerun_power)
                    rerun_cmd_none = _shlex.join(rerun_none)
                    rerun_script_path = Path(evidence_dir) / "suggested_rerun_commands.sh"
                    try:
                        rerun_script = (
                            "#!/usr/bin/env bash\n"
                            "set -euo pipefail\n\n"
                            "# Conservative rerun (power calibration)\n"
                            f"{rerun_cmd_power}\n\n"
                            "# Minimal-complexity rerun (no calibration)\n"
                            f"{rerun_cmd_none}\n"
                        )
                        rerun_script_path.write_text(rerun_script, encoding="utf-8")
                        rerun_script_path.chmod(0o755)
                        rerun_label = "复跑脚本" if LANG == "zh" else "Rerun script"
                        readiness_lines.append(f"  {rerun_label:<16} {rerun_script_path}")
                    except Exception:
                        pass
                print()
                box(t("r_quick_readiness"), readiness_lines, color="C")
                if set(blockers) & calibration_blockers:
                    print()
                    if LANG == "zh":
                        print(f"  {s('C', '可直接复制的复跑命令：')}")
                    else:
                        print(f"  {s('C', 'Copy-ready rerun commands:')}")
                    print(f"  {s('W', rerun_cmd_power)}")
                    print(f"  {s('W', rerun_cmd_none)}")

            # Model selection summary (mean±std from CV)
            ms_path = Path(evidence_dir) / "model_selection_report.json"
            if ms_path.exists():
                ms_data = _json.loads(ms_path.read_text())
                candidates = ms_data.get("candidates", [])
                sel_id = ms_data.get("selected_model_id", "")
                trace = ms_data.get("selection_trace", {})
                if candidates:
                    sel_lines = []
                    sel_lines.append(f"  {'Candidates':<14} {len(candidates)}")
                    if trace.get("one_se_threshold") is not None:
                        sel_lines.append(f"  {'1-SE cutoff':<14} {float(trace['one_se_threshold']):.4f}")
                    sel_lines.append("")
                    sel_lines.append(f"  {'':20} {'mean':>8}  {'std':>8}  {'folds':>5}")
                    top = sorted(candidates,
                                 key=lambda c: -float(c.get("selection_metrics", {}).get("pr_auc", {}).get("mean", 0)))
                    shown = list(top[:5])
                    if sel_id and all(str(c.get("model_id")) != str(sel_id) for c in shown):
                        selected_row = next((c for c in top if str(c.get("model_id")) == str(sel_id)), None)
                        if selected_row is not None:
                            shown.append(selected_row)
                    for c in shown:
                        sm = c.get("selection_metrics", {}).get("pr_auc", {})
                        m = sm.get("mean")
                        sd = sm.get("std")
                        nf = sm.get("n_folds", 1)
                        if m is not None:
                            tag = " *" if c.get("model_id") == sel_id else ""
                            name = str(c.get("model_id", "?"))[:20]
                            sd_str = f"{float(sd):.4f}" if sd else "   --"
                            sel_lines.append(f"  {name:<20} {float(m):>8.4f}  {sd_str:>8}  {nf:>5}{tag}")
                    sel_lines.append("")
                    sel_lines.append(f"  {DIM}* = selected (1-SE rule){RST}")
                    print()
                    box("Model Selection (CV)", sel_lines, color="C")
        else:
            readiness_error = "evaluation_report_missing"
    except Exception:
        readiness_error = "evaluation_report_parse_error"

    state["_play_readiness_blockers"] = list(play_blockers)
    state["_play_readiness_advisories"] = list(play_advisories)
    if not readiness_evaluated and readiness_error:
        play_advisories.append("quick_readiness_unavailable")
        state["_play_readiness_advisories"] = list(play_advisories)
    state["_play_readiness_evaluated"] = bool(readiness_evaluated)
    if readiness_error:
        state["_play_readiness_error"] = readiness_error
    else:
        state.pop("_play_readiness_error", None)
    if bool(state.get("_fail_on_play_blockers", False)) and not readiness_evaluated:
        print(f"\n  {s('R', t('r_play_readiness_unavailable'))}")
        reason_label = t("r_play_readiness_reason")
        print(f"  {reason_label} {readiness_error or 'unknown'}")
        state.pop("_from_history", None)
        _save_history(state)
        return FAIL
    if bool(state.get("_fail_on_play_blockers", False)) and play_blockers:
        print(f"\n  {s('R', t('r_play_blocking_fail'))}")
        for item in play_blockers[:6]:
            print(f"  - {item}")
        state.pop("_from_history", None)
        _save_history(state)
        return FAIL

    # Save run to history
    state.pop("_from_history", None)
    _save_history(state)
    print(f"\n  {DIM}{t('r_next')}{RST}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WIZARD
# ══════════════════════════════════════════════════════════════════════════════

def wizard(
    force_lang: str = "",
    dry_run: bool = False,
    strict_small_sample: bool = False,
    strict_small_sample_max_rows: int = STRICT_SMALL_SAMPLE_DEFAULT_MAX_ROWS,
    fail_on_play_blockers: bool = False,
) -> int:
    global LANG
    LANG = detect_lang()

    state: Dict[str, Any] = {
        "_dry_run": dry_run,
        "_strict_small_sample": bool(strict_small_sample),
        "_strict_small_sample_max_rows": int(strict_small_sample_max_rows),
        "_fail_on_play_blockers": bool(fail_on_play_blockers),
    }
    steps = [step_lang, step_source, step_dataset, step_config,
             step_split, step_imbalance, step_models, step_tuning,
             step_advanced, step_confirm, step_run]
    skipped: set = set()
    i = 0

    if force_lang in ("en", "zh"):
        LANG = force_lang
        state["lang"] = LANG
        skipped.add(0)
        i = 1

    while i < len(steps):
        try:
            result = steps[i](state)
        except KeyboardInterrupt:
            print(f"\n  {DIM}{t('interrupted')}{RST}")
            return 130

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
        elif result is FAIL:
            return 2
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
    import argparse as _ap
    parser = _ap.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--lang", choices=["en", "zh"], default="",
                        help="Set language directly, skipping the language selection step.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print commands without executing them.")
    parser.add_argument(
        "--strict-small-sample",
        action="store_true",
        default=False,
        help=(
            "Enable strict small-sample profile (linear-only model pool, capped trials, "
            "conservative calibration) when rows <= threshold."
        ),
    )
    parser.add_argument(
        "--strict-small-sample-max-rows",
        type=int,
        default=STRICT_SMALL_SAMPLE_DEFAULT_MAX_ROWS,
        help=f"Row-count threshold for --strict-small-sample (default: {STRICT_SMALL_SAMPLE_DEFAULT_MAX_ROWS}).",
    )
    parser.add_argument(
        "--fail-on-play-blockers",
        action="store_true",
        default=False,
        help=(
            "Return non-zero when play quick-readiness contains blocking items "
            "(not a replacement for workflow --strict)."
        ),
    )
    args, _ = parser.parse_known_args()
    if int(args.strict_small_sample_max_rows) < 1:
        raise SystemExit("--strict-small-sample-max-rows must be >= 1.")
    return wizard(
        force_lang=args.lang,
        dry_run=args.dry_run,
        strict_small_sample=bool(args.strict_small_sample),
        strict_small_sample_max_rows=int(args.strict_small_sample_max_rows),
        fail_on_play_blockers=bool(args.fail_on_play_blockers),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        sys.stdout.write(SHOW_CUR); sys.stdout.flush()
