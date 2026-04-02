"""
F1 Strategy Pipeline — Web Server
===================================
Bachelor's Thesis Tool — LUT University

Serves the index.html dashboard and exposes API endpoints
that trigger the Python pipeline and stream results back live.

Usage:
    pip install flask
    python server.py
    → open http://localhost:5001

Endpoints:
    GET  /                      — serves index.html
    GET  /api/status            — pipeline status + output file list
    POST /api/run               — start full pipeline (analyser → rf → f1)
    POST /api/run/season        — simulate a single season
    GET  /api/stream            — SSE stream of live pipeline log output
    GET  /api/results           — sweep CSV as JSON for the dashboard
    GET  /outputs/<filename>    — serve output PNGs and CSVs
"""

from flask import (
    Flask, request, jsonify, send_from_directory,
    Response, stream_with_context
)
import subprocess, threading, queue, os, sys, json, csv
from datetime import datetime

_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(_HERE, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=_HERE)

# ── Pipeline state (one run at a time) ────────────────────────────────────────
_pipeline_lock    = threading.Lock()
_pipeline_running = False
_log_queue        = queue.Queue()
_last_run         = None


def _stream_subprocess(cmd):
    """
    Run cmd as subprocess, push every line into _log_queue.
    Browser reads lines via /api/stream (SSE).
    """
    global _pipeline_running, _last_run

    with _pipeline_lock:
        if _pipeline_running:
            _log_queue.put("__ERROR__ Pipeline already running")
            return
        _pipeline_running = True

    _last_run = datetime.now().isoformat()
    _log_queue.put(f"__START__ {datetime.now().strftime('%H:%M:%S')}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=_HERE,
        )
        for line in proc.stdout:
            _log_queue.put(line.rstrip())
        proc.wait()
        if proc.returncode == 0:
            _log_queue.put("__DONE__")
        else:
            _log_queue.put(f"__ERROR__ Process exited with code {proc.returncode}")
    except Exception as exc:
        _log_queue.put(f"__ERROR__ {exc}")
    finally:
        _pipeline_running = False


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(_HERE, "index.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/api/status")
def status():
    files = []
    for fname in sorted(os.listdir(OUTPUTS_DIR)):
        fpath = os.path.join(OUTPUTS_DIR, fname)
        if os.path.isfile(fpath):
            files.append({
                "name":     fname,
                "size_kb":  round(os.path.getsize(fpath) / 1024, 1),
                "modified": datetime.fromtimestamp(
                    os.path.getmtime(fpath)
                ).strftime("%Y-%m-%d %H:%M"),
            })
    return jsonify({
        "running":  _pipeline_running,
        "last_run": _last_run,
        "outputs":  files,
    })


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """Start the full pipeline: analyser → rf → f1."""
    if _pipeline_running:
        return jsonify({"error": "Pipeline already running"}), 409

    body  = request.get_json(silent=True) or {}
    start = body.get("start", 1994)
    end   = body.get("end",   2024)

    # Always headless on server — PNGs are saved to outputs/ and served
    cmd = [
        sys.executable,
        os.path.join(_HERE, "analyser.py"),
        "--start", str(start),
        "--end",   str(end),
        "--no-plot",
    ]

    threading.Thread(target=_stream_subprocess, args=(cmd,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/run/season", methods=["POST"])
def run_season():
    """Simulate a single season from an existing dataset."""
    if _pipeline_running:
        return jsonify({"error": "Pipeline already running"}), 409

    body = request.get_json(silent=True) or {}
    year = body.get("year", 2010)

    dataset_path = os.path.join(OUTPUTS_DIR, "f1_rf_dataset.csv")
    if not os.path.exists(dataset_path):
        return jsonify({
            "error": "No dataset found — run the full pipeline first"
        }), 400

    cmd = [
        sys.executable,
        os.path.join(_HERE, "f1.py"),
        "--year",    str(year),
        "--dataset", dataset_path,
        "--outputs", OUTPUTS_DIR,
        "--no-plot",
    ]

    threading.Thread(target=_stream_subprocess, args=(cmd,), daemon=True).start()
    return jsonify({"status": "started", "year": year})


@app.route("/api/stream")
def stream():
    """
    Server-Sent Events — browser connects and receives live log lines.
    Each message: data: "<json-encoded line>"
    Special tokens: __START__  __DONE__  __ERROR__  __HEARTBEAT__
    """
    def event_stream():
        while True:
            try:
                line = _log_queue.get(timeout=25)
                yield f"data: {json.dumps(line)}\n\n"
                if line.startswith("__DONE__") or line.startswith("__ERROR__"):
                    break
            except queue.Empty:
                yield 'data: "__HEARTBEAT__"\n\n'

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/results")
def results():
    """Return sweep CSV + metadata as JSON for the live dashboard."""
    sweep_path   = os.path.join(OUTPUTS_DIR, "f1_simulation_sweep.csv")
    dataset_path = os.path.join(OUTPUTS_DIR, "f1_rf_dataset.csv")

    out = {
        "sweep":        [],
        "dataset_rows": 0,
        "images":       {},
    }

    if os.path.exists(sweep_path):
        with open(sweep_path, newline="") as f:
            for row in csv.DictReader(f):
                for k in ["year", "avg_variance_s", "avg_points_gain"]:
                    if k in row:
                        try:
                            row[k] = float(row[k])
                        except ValueError:
                            pass
                row["title_flips"] = row.get("title_flips", "False") == "True"
                out["sweep"].append(row)

    if os.path.exists(dataset_path):
        with open(dataset_path) as f:
            out["dataset_rows"] = sum(1 for _ in f) - 1

    for img in ["f1_rf_results.png", "f1_sweep_simulation.png"]:
        out["images"][img] = os.path.exists(os.path.join(OUTPUTS_DIR, img))

    return jsonify(out)


if __name__ == "__main__":
    print("\n  F1 Strategy Pipeline — Web Server")
    print("  LUT University — Bachelor's Thesis")
    print(f"\n  Project root: {_HERE}")
    print(f"  Outputs:      {OUTPUTS_DIR}")
    print("\n  Open → http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)