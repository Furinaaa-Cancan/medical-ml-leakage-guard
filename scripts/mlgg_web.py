#!/usr/bin/env python3
"""
ML Leakage Guard — Local Web UI Wizard.

A lightweight Flask-based web interface that mirrors the 9-step
interactive wizard (mlgg_pixel.py) in a browser.

Binds to 127.0.0.1:8501 (localhost only, never exposed externally).

Usage:
    python3 scripts/mlgg_web.py
    # Then open http://127.0.0.1:8501 in a browser.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from flask import (
        Flask,
        Response,
        jsonify,
        redirect,
        render_template_string,
        request,
        send_from_directory,
        url_for,
    )
except ImportError:
    print(
        "Flask is required: pip install flask",
        file=sys.stderr,
    )
    raise SystemExit(1)

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="mlgg_web_"))
PYTHON = sys.executable

app = Flask(__name__)
app.secret_key = os.urandom(32)

# ── session state ──────────────────────────────────────────────────────────────
_sessions: Dict[str, Dict[str, Any]] = {}
_log_queues: Dict[str, queue.Queue] = {}


def get_session(sid: str) -> Dict[str, Any]:
    """Get or create a session state dict."""
    if sid not in _sessions:
        _sessions[sid] = {
            "step": 1,
            "project_root": "",
            "csv_path": "",
            "target_col": "y",
            "patient_id_col": "patient_id",
            "time_col": "event_time",
            "model_pool": "logistic_l1,logistic_l2,random_forest_balanced",
            "cv_splits": 5,
            "running": False,
            "result": None,
        }
    return _sessions[sid]


# ── HTML template ──────────────────────────────────────────────────────────────
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Leakage Guard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 720px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.5rem; margin-bottom: 1rem; color: #38bdf8; }
  h2 { font-size: 1.1rem; margin-bottom: 0.5rem; color: #94a3b8; }

  /* Progress bar */
  .progress { display: flex; gap: 4px; margin-bottom: 2rem; }
  .progress .step { flex: 1; height: 6px; border-radius: 3px; background: #1e293b; }
  .progress .step.done { background: #22c55e; }
  .progress .step.active { background: #38bdf8; }

  /* Form */
  .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
  label { display: block; margin-bottom: 0.25rem; font-size: 0.85rem; color: #94a3b8; }
  input, select, textarea { width: 100%; padding: 0.5rem; border: 1px solid #334155;
    border-radius: 6px; background: #0f172a; color: #e2e8f0; font-size: 0.9rem;
    margin-bottom: 0.75rem; }
  input[type="file"] { padding: 0.25rem; }
  button { padding: 0.6rem 1.5rem; border: none; border-radius: 6px; cursor: pointer;
    font-size: 0.9rem; font-weight: 600; }
  .btn-primary { background: #2563eb; color: white; }
  .btn-primary:hover { background: #1d4ed8; }
  .btn-secondary { background: #334155; color: #e2e8f0; }
  .btn-danger { background: #dc2626; color: white; }
  .btn-group { display: flex; gap: 0.5rem; margin-top: 1rem; }

  /* Log area */
  #log { background: #0f172a; border: 1px solid #334155; border-radius: 6px;
    padding: 0.75rem; font-family: monospace; font-size: 0.8rem;
    height: 300px; overflow-y: auto; white-space: pre-wrap; color: #a3e635; }

  /* Result */
  .result-pass { color: #22c55e; font-size: 1.2rem; font-weight: bold; }
  .result-fail { color: #ef4444; font-size: 1.2rem; font-weight: bold; }
  .info { color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem; }
</style>
</head>
<body>
<div class="container">
  <h1>ML Leakage Guard</h1>

  <!-- Progress -->
  <div class="progress">
    {% for i in range(1, 10) %}
    <div class="step {% if i < step %}done{% elif i == step %}active{% endif %}"></div>
    {% endfor %}
  </div>

  {% if step == 1 %}
  <div class="card">
    <h2>Step 1 — Project Setup</h2>
    <form method="post" action="/step/1">
      <input type="hidden" name="sid" value="{{ sid }}">
      <label>Project Root Directory</label>
      <input name="project_root" value="{{ session.project_root or '' }}"
             placeholder="/path/to/project">
      <div class="btn-group">
        <button type="submit" class="btn-primary">Next &rarr;</button>
      </div>
    </form>
  </div>

  {% elif step == 2 %}
  <div class="card">
    <h2>Step 2 — Upload CSV Data</h2>
    <form method="post" action="/step/2" enctype="multipart/form-data">
      <input type="hidden" name="sid" value="{{ sid }}">
      <label>CSV File</label>
      <input type="file" name="csv_file" accept=".csv">
      <label>Or enter path</label>
      <input name="csv_path" value="{{ session.csv_path or '' }}" placeholder="/path/to/data.csv">
      <div class="btn-group">
        <button type="submit" class="btn-primary">Next &rarr;</button>
      </div>
    </form>
  </div>

  {% elif step == 3 %}
  <div class="card">
    <h2>Step 3 — Column Configuration</h2>
    <form method="post" action="/step/3">
      <input type="hidden" name="sid" value="{{ sid }}">
      <label>Target Column</label>
      <input name="target_col" value="{{ session.target_col }}">
      <label>Patient ID Column</label>
      <input name="patient_id_col" value="{{ session.patient_id_col }}">
      <label>Time Column</label>
      <input name="time_col" value="{{ session.time_col }}">
      <div class="btn-group">
        <button type="submit" class="btn-primary">Next &rarr;</button>
      </div>
    </form>
  </div>

  {% elif step == 4 %}
  <div class="card">
    <h2>Step 4 — Model Pool</h2>
    <form method="post" action="/step/4">
      <input type="hidden" name="sid" value="{{ sid }}">
      <label>Model Families (comma-separated)</label>
      <input name="model_pool" value="{{ session.model_pool }}">
      <label>CV Splits</label>
      <input name="cv_splits" type="number" value="{{ session.cv_splits }}" min="3" max="20">
      <div class="btn-group">
        <button type="submit" class="btn-primary">Next &rarr;</button>
      </div>
    </form>
  </div>

  {% elif step == 5 %}
  <div class="card">
    <h2>Step 5 — Review Configuration</h2>
    <pre style="color: #94a3b8; font-size: 0.85rem;">
Project:    {{ session.project_root }}
CSV:        {{ session.csv_path }}
Target:     {{ session.target_col }}
Patient ID: {{ session.patient_id_col }}
Time Col:   {{ session.time_col }}
Models:     {{ session.model_pool }}
CV Splits:  {{ session.cv_splits }}
    </pre>
    <form method="post" action="/step/5">
      <input type="hidden" name="sid" value="{{ sid }}">
      <div class="btn-group">
        <button type="submit" class="btn-primary">Confirm &amp; Split Data &rarr;</button>
        <a href="/reset?sid={{ sid }}" class="btn-secondary"
           style="text-decoration:none; text-align:center; line-height:2;">Start Over</a>
      </div>
    </form>
  </div>

  {% elif step == 6 %}
  <div class="card">
    <h2>Step 6 — Data Splitting</h2>
    <div id="log">Waiting for logs...</div>
    <div class="btn-group">
      <button id="btn-next" class="btn-primary" onclick="location.href='/advance?sid={{ sid }}'"
              style="display:none">Next &rarr;</button>
    </div>
  </div>
  <script>
    const es = new EventSource("/logs/{{ sid }}");
    const el = document.getElementById("log");
    es.onmessage = function(e) {
      if (e.data === "__DONE__") { es.close(); document.getElementById("btn-next").style.display=""; return; }
      el.textContent += e.data + "\n";
      el.scrollTop = el.scrollHeight;
    };
  </script>

  {% elif step == 7 %}
  <div class="card">
    <h2>Step 7 — Training &amp; Evaluation</h2>
    <div id="log">Waiting for logs...</div>
    <div class="btn-group">
      <button id="btn-next" class="btn-primary" onclick="location.href='/advance?sid={{ sid }}'"
              style="display:none">Next &rarr;</button>
    </div>
  </div>
  <script>
    const es = new EventSource("/logs/{{ sid }}");
    const el = document.getElementById("log");
    es.onmessage = function(e) {
      if (e.data === "__DONE__") { es.close(); document.getElementById("btn-next").style.display=""; return; }
      el.textContent += e.data + "\n";
      el.scrollTop = el.scrollHeight;
    };
  </script>

  {% elif step == 8 %}
  <div class="card">
    <h2>Step 8 — Gate Pipeline</h2>
    <div id="log">Waiting for logs...</div>
    <div class="btn-group">
      <button id="btn-next" class="btn-primary" onclick="location.href='/advance?sid={{ sid }}'"
              style="display:none">Next &rarr;</button>
    </div>
  </div>
  <script>
    const es = new EventSource("/logs/{{ sid }}");
    const el = document.getElementById("log");
    es.onmessage = function(e) {
      if (e.data === "__DONE__") { es.close(); document.getElementById("btn-next").style.display=""; return; }
      el.textContent += e.data + "\n";
      el.scrollTop = el.scrollHeight;
    };
  </script>

  {% elif step == 9 %}
  <div class="card">
    <h2>Step 9 — Results</h2>
    {% if session.result %}
      {% if session.result.status == 'pass' %}
        <p class="result-pass">PASS — Publication-Grade Claim Verified</p>
      {% else %}
        <p class="result-fail">FAIL — Claim Blocked</p>
      {% endif %}
      <pre style="color: #94a3b8; font-size: 0.8rem; max-height: 400px; overflow-y: auto;">{{ session.result | tojson(indent=2) }}</pre>
    {% else %}
      <p>Pipeline complete. Check evidence/ for reports.</p>
    {% endif %}
    <div class="btn-group">
      <a href="/reset?sid={{ sid }}" class="btn-primary"
         style="text-decoration:none; text-align:center;">New Run</a>
    </div>
  </div>
  {% endif %}

  <p class="info">Bound to 127.0.0.1:8501 &mdash; not externally accessible.</p>
</div>
</body>
</html>"""


# ── routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main wizard page for the current session."""
    sid = request.cookies.get("sid") or str(uuid.uuid4())
    session = get_session(sid)
    resp = Response(
        render_template_string(
            PAGE_HTML, step=session["step"], session=session, sid=sid
        )
    )
    resp.set_cookie("sid", sid, httponly=True, samesite="Strict")
    return resp


@app.route("/step/<int:step_num>", methods=["POST"])
def handle_step(step_num: int):
    """Process form submission for a wizard step."""
    sid = request.form.get("sid") or request.cookies.get("sid", "")
    session = get_session(sid)

    if step_num == 1:
        raw = request.form.get("project_root", "").strip()
        if not raw:
            return "Project root is required.", 400
        p = Path(raw).expanduser().resolve()
        if ".." in raw:
            return "Path traversal not allowed.", 400
        session["project_root"] = str(p)
        session["step"] = 2

    elif step_num == 2:
        csv_file = request.files.get("csv_file")
        csv_path = request.form.get("csv_path", "").strip()
        if csv_file and csv_file.filename:
            safe_name = Path(csv_file.filename).name
            if ".." in safe_name or "/" in safe_name:
                return "Invalid filename.", 400
            dest = UPLOAD_DIR / f"{sid}_{safe_name}"
            csv_file.save(str(dest))
            session["csv_path"] = str(dest)
        elif csv_path:
            if ".." in csv_path:
                return "Path traversal not allowed.", 400
            session["csv_path"] = csv_path
        else:
            return "CSV file or path required.", 400
        session["step"] = 3

    elif step_num == 3:
        session["target_col"] = request.form.get("target_col", "y").strip()
        session["patient_id_col"] = request.form.get("patient_id_col", "patient_id").strip()
        session["time_col"] = request.form.get("time_col", "event_time").strip()
        session["step"] = 4

    elif step_num == 4:
        session["model_pool"] = request.form.get("model_pool", "").strip()
        try:
            cv = int(request.form.get("cv_splits", 5))
            if cv < 3:
                cv = 3
        except ValueError:
            cv = 5
        session["cv_splits"] = cv
        session["step"] = 5

    elif step_num == 5:
        session["step"] = 6
        _start_split(sid, session)

    return redirect(url_for("index"))


@app.route("/advance")
def advance():
    """Advance to the next pipeline phase after a background task completes."""
    sid = request.args.get("sid") or request.cookies.get("sid", "")
    session = get_session(sid)
    if session["step"] == 6:
        session["step"] = 7
        _start_train(sid, session)
    elif session["step"] == 7:
        session["step"] = 8
        _start_pipeline(sid, session)
    elif session["step"] == 8:
        session["step"] = 9
    return redirect(url_for("index"))


@app.route("/reset")
def reset():
    """Reset the session and start a new wizard run."""
    sid = request.args.get("sid") or request.cookies.get("sid", "")
    if sid in _sessions:
        del _sessions[sid]
    return redirect(url_for("index"))


@app.route("/logs/<sid>")
def stream_logs(sid: str):
    """SSE endpoint for real-time log streaming."""
    def generate():
        q = _log_queues.get(sid)
        if q is None:
            yield "data: No active process.\n\n"
            yield "data: __DONE__\n\n"
            return
        while True:
            try:
                line = q.get(timeout=30)
                if line is None:
                    yield "data: __DONE__\n\n"
                    return
                yield f"data: {line}\n\n"
            except queue.Empty:
                yield "data: [waiting...]\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ── background tasks ──────────────────────────────────────────────────────────
def _run_cmd_with_logs(sid: str, cmd: List[str], cwd: str) -> int:
    """Run a command, streaming stdout/stderr to the session log queue."""
    q: queue.Queue = queue.Queue()
    _log_queues[sid] = q
    q.put(f"$ {shlex.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        q.put(line.rstrip())
    proc.wait()
    q.put(f"\n[exit code: {proc.returncode}]")
    q.put(None)  # sentinel
    return proc.returncode


def _start_split(sid: str, session: Dict[str, Any]) -> None:
    """Start data splitting in a background thread."""
    project = session["project_root"]
    csv_path = session["csv_path"]
    cmd = [
        PYTHON,
        str(SCRIPTS_DIR / "split_data.py"),
        "--input", csv_path,
        "--output-dir", str(Path(project) / "data"),
        "--patient-id-col", session["patient_id_col"],
        "--target-col", session["target_col"],
        "--time-col", session["time_col"],
    ]
    threading.Thread(
        target=_run_cmd_with_logs, args=(sid, cmd, project), daemon=True
    ).start()


def _start_train(sid: str, session: Dict[str, Any]) -> None:
    """Start training in a background thread."""
    project = session["project_root"]
    data_dir = Path(project) / "data"
    evidence_dir = Path(project) / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(SCRIPTS_DIR / "train_select_evaluate.py"),
        "--train", str(data_dir / "train.csv"),
        "--test", str(data_dir / "test.csv"),
        "--target-col", session["target_col"],
        "--patient-id-col", session["patient_id_col"],
        "--ignore-cols", f"{session['patient_id_col']},{session['time_col']}",
        "--model-pool", session["model_pool"],
        "--cv-splits", str(session["cv_splits"]),
        "--model-selection-report-out", str(evidence_dir / "model_selection_report.json"),
        "--evaluation-report-out", str(evidence_dir / "evaluation_report.json"),
    ]
    valid_csv = data_dir / "valid.csv"
    if valid_csv.exists():
        cmd.extend(["--valid", str(valid_csv)])
    threading.Thread(
        target=_run_cmd_with_logs, args=(sid, cmd, project), daemon=True
    ).start()


def _start_pipeline(sid: str, session: Dict[str, Any]) -> None:
    """Start gate pipeline in a background thread."""
    project = session["project_root"]
    evidence_dir = Path(project) / "evidence"

    def _run():
        request_json = evidence_dir / "request.json"
        if not request_json.exists():
            q = _log_queues.get(sid) or queue.Queue()
            _log_queues[sid] = q
            q.put("[SKIP] No request.json found — pipeline requires onboarding first.")
            q.put(None)
            return
        cmd = [
            PYTHON,
            str(SCRIPTS_DIR / "run_strict_pipeline.py"),
            "--request", str(request_json),
            "--evidence-dir", str(evidence_dir),
            "--strict",
            "--allow-missing-compare",
            "--report", str(evidence_dir / "strict_pipeline_report.json"),
        ]
        _run_cmd_with_logs(sid, cmd, project)

        report_path = evidence_dir / "strict_pipeline_report.json"
        if report_path.exists():
            try:
                session["result"] = json.loads(report_path.read_text())
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    """Start the local web UI server."""
    print("ML Leakage Guard Web UI")
    print(f"Open http://127.0.0.1:8501 in your browser.")
    print(f"Press Ctrl+C to stop.\n")
    try:
        app.run(host="127.0.0.1", port=8501, debug=False, threaded=True)
    except OSError as exc:
        if "Address already in use" in str(exc):
            print(f"Port 8501 is already in use. Try: lsof -i :8501", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
