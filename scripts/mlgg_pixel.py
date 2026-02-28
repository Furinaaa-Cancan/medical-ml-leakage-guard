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
TOTAL_STEPS = 6

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
    "bye":           {"en": "Bye!", "zh": "\u518d\u89c1\uff01"},
    "interrupted":   {"en": "Interrupted.", "zh": "\u5df2\u4e2d\u65ad\u3002"},
    "enter_continue":{"en": "Press Enter to continue...",
                      "zh": "\u6309 Enter \u7ee7\u7eed..."},

    "s_lang":        {"en": "Language", "zh": "\u8bed\u8a00"},
    "s_source":      {"en": "Data Source", "zh": "\u6570\u636e\u6765\u6e90"},
    "s_dataset":     {"en": "Dataset", "zh": "\u6570\u636e\u96c6"},
    "s_config":      {"en": "Configure", "zh": "\u914d\u7f6e"},
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

    "c_file":        {"en": "File:", "zh": "\u6587\u4ef6\uff1a"},
    "c_pid":         {"en": "Patient ID:", "zh": "\u60a3\u8005 ID\uff1a"},
    "c_target":      {"en": "Target:", "zh": "\u76ee\u6807\uff1a"},
    "c_time":        {"en": "Time:", "zh": "\u65f6\u95f4\uff1a"},
    "c_strat":       {"en": "Strategy:", "zh": "\u7b56\u7565\uff1a"},
    "c_output":      {"en": "Output:", "zh": "\u8f93\u51fa\uff1a"},
    "c_none":        {"en": "(none)", "zh": "(\u65e0)"},
    "c_start":       {"en": "Start Pipeline", "zh": "\u5f00\u59cb\u8fd0\u884c"},
    "c_back":        {"en": "Go Back", "zh": "\u8fd4\u56de\u4fee\u6539"},

    "x_download":    {"en": "Downloading {ds}...", "zh": "\u6b63\u5728\u4e0b\u8f7d {ds}..."},
    "x_split":       {"en": "Splitting with safety checks...",
                      "zh": "\u6b63\u5728\u5b89\u5168\u5206\u5272..."},
    "x_pipeline":    {"en": "Running full pipeline...",
                      "zh": "\u6b63\u5728\u8fd0\u884c\u5b8c\u6574\u7ba1\u7ebf..."},
    "x_fail":        {"en": "Failed.", "zh": "\u5931\u8d25\u3002"},

    "r_done":        {"en": "Complete!", "zh": "\u5b8c\u6210\uff01"},
    "r_split_ok":    {"en": "Split Complete!", "zh": "\u5206\u5272\u5b8c\u6210\uff01"},
    "r_saved":       {"en": "Saved to:", "zh": "\u4fdd\u5b58\u81f3\uff1a"},
    "r_next":        {"en": "Data ready. Re-run wizard and choose Demo to train.",
                      "zh": "\u6570\u636e\u5df2\u5c31\u7eea\u3002\u91cd\u65b0\u8fd0\u884c\u5e76\u9009\u62e9\u6f14\u793a\u6a21\u5f0f\u53ef\u5f00\u59cb\u8bad\u7ec3\u3002"},

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
            if ch == b"q": return "Q"
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
        state["strategy"] = "grouped_temporal"
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

    # Patient ID -- put auto-detected first
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

    # Strategy
    _config_header((t("c_pid"), pid), (t("c_target"), tgt))
    si = select(
        [t("strat_temporal"), t("strat_random"), t("strat_stratified")],
        [t("strat_temporal_d"), t("strat_random_d"), t("strat_stratified_d")],
        title=t("pick_strat"),
    )
    if si < 0: return BACK
    strat = ["grouped_temporal", "grouped_random", "stratified_grouped"][si]

    # Time column
    tcol = ""
    if strat == "grouped_temporal":
        _config_header((t("c_pid"), pid), (t("c_target"), tgt), (t("c_strat"), strat))
        rem2 = [c for c in columns if c not in (pid, tgt)]
        if not rem2:
            print(f"  {s('R', t('no_time_col'))}")
            sys.stdout.write(SHOW_CUR)
            try:
                input(f"  {DIM}{t('enter_continue')}{RST}")
            except (EOFError, KeyboardInterrupt):
                pass
            return BACK
        time_opts = rem2[:]
        if detected["time"] and detected["time"] in time_opts:
            time_opts.remove(detected["time"])
            time_opts.insert(0, detected["time"])
        tci = select(time_opts, title=t("pick_time"))
        if tci < 0: return BACK
        tcol = time_opts[tci]

    state["pid"] = pid
    state["target"] = tgt
    state["strategy"] = strat
    state["time"] = tcol
    return True


def step_confirm(state: Dict) -> Any:
    _clear()
    step_header(5, TOTAL_STEPS, t("s_confirm"))

    if state["dataset_key"] == "demo":
        box("Demo Pipeline", [
            f"{t('c_output')} {state['out_dir']}/",
            "",
            "Synthetic data | 28 safety gates | ~5 min",
        ], color="C")
    else:
        fname = Path(state["csv_path"]).name if state.get("csv_path") else "?"
        box(t("s_confirm"), [
            f"{t('c_file')}    {fname}",
            f"{t('c_pid')}     {state.get('pid', '?')}",
            f"{t('c_target')}  {state.get('target', '?')}",
            f"{t('c_time')}    {state.get('time') or t('c_none')}",
            f"{t('c_strat')}   {state.get('strategy', '?')}",
            f"{t('c_output')}  {state['out_dir']}/",
        ], color="C")

    print()
    ci = select([t("c_start"), t("c_back")])
    if ci != 0:
        return BACK
    return True


def step_run(state: Dict) -> Any:
    _clear()
    step_header(6, TOTAL_STEPS, t("s_run"))

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

    # ── Download + Split flow ──
    steps_list: List[Tuple[str, str]] = []
    if source == "download":
        ds_names = {"heart": t("ds_heart"), "breast": t("ds_breast"), "ckd": t("ds_kidney")}
        ds_label = ds_names.get(state.get("dataset_key", ""), state.get("dataset_key", ""))
        steps_list.append((t("x_download", ds=ds_label), "pending"))
    steps_list.append((t("x_split"), "pending"))

    print()
    render_steps(steps_list)
    step_idx = 0

    # Download if needed
    if source == "download":
        erase_n(len(steps_list))
        steps_list[0] = (steps_list[0][0], "running")
        render_steps(steps_list)

        rc, _, err = run_spinner(
            [sys.executable, str(EXAMPLES_DIR / "download_real_data.py"),
             state["dataset_key"]],
            steps_list[0][0],
        )
        erase_n(len(steps_list))
        if rc != 0:
            steps_list[0] = (steps_list[0][0], "fail")
            render_steps(steps_list)
            print(f"\n  {s('R', t('x_fail'))}")
            return True
        steps_list[0] = (steps_list[0][0], "done")
        step_idx = 1

    # Split
    erase_n(len(steps_list))
    steps_list[step_idx] = (steps_list[step_idx][0], "running")
    render_steps(steps_list)

    out_data = str(Path(state["out_dir"]) / "data")
    csv_path = state["csv_path"]

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "mlgg.py"), "split", "--",
        "--input", csv_path, "--output-dir", out_data,
        "--patient-id-col", state["pid"], "--target-col", state["target"],
        "--strategy", state["strategy"],
    ]
    if state.get("time"):
        cmd.extend(["--time-col", state["time"]])

    rc, _, err = run_spinner(cmd, steps_list[step_idx][0])
    erase_n(len(steps_list))

    if rc != 0:
        steps_list[step_idx] = (steps_list[step_idx][0], "fail")
        render_steps(steps_list)
        print(f"\n  {s('R', t('x_fail'))}")
        if err:
            for l in err.strip().split("\n")[-3:]:
                print(f"  {DIM}{l}{RST}")
        return True

    steps_list[step_idx] = (steps_list[step_idx][0], "done")
    render_steps(steps_list)

    # Show results
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
            "", f"{t('r_saved')} {out_data}/",
        ], color="G")
    except Exception:
        print(f"\n  {s('G', '[ok]', True)} {out_data}/")

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
             step_confirm, step_run]
    skipped: set = set()  # steps that auto-skipped (no user interaction)
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
            # Skip over auto-skipped steps when going backward
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
    # Flush any keys buffered during spinner execution
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
