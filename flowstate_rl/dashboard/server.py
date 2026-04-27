"""
server.py
---------
FlowState-RL dashboard server.

Endpoints
---------
  GET /         → serves dashboard/index.html
  GET /metrics  → returns data/live_metrics.json  (or {} if file absent)

Usage
-----
  pip install flask flask-cors
  python -m flowstate_rl.dashboard.server

  Open: http://localhost:5000
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, send_file
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve().parent          # flowstate_rl/dashboard/
_PROJECT   = _HERE.parents[1]                         # project root
_INDEX     = _HERE / "index.html"
_METRICS   = _PROJECT / "data" / "live_metrics.json"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=str(_HERE), static_url_path="")
CORS(app)   # enable CORS for all routes and origins


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the dashboard HTML page."""
    return send_file(_INDEX)


@app.route("/metrics")
def metrics():
    """
    Return the latest metrics as JSON.

    Reads data/live_metrics.json if it exists; returns an empty object
    otherwise so the dashboard can handle the no-data case gracefully.
    """
    if _METRICS.exists():
        try:
            with open(_METRICS, encoding="utf-8") as f:
                data = json.load(f)
            if data: # ensure not empty
                return jsonify(data)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    return jsonify({"status": "waiting"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("\nFlowState server running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
