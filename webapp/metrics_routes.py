"""Serve the real evaluation report to the front end.

Paste these routes into app.py (or import this module and call
`register_metrics(app, outputs_dir)`).

Every number the web UI displays comes from outputs/eval_report.json, which is
written by `python -m src.evaluate`. Nothing is hardcoded in the page. If the
report is missing, the UI shows "not measured yet" rather than a placeholder
figure.
"""
from __future__ import annotations

import json
import os

from flask import jsonify


def register_metrics(app, outputs_dir: str = "outputs"):
    report_path = os.path.join(outputs_dir, "eval_report.json")

    def _load():
        if not os.path.exists(report_path):
            return None
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @app.route("/api/metrics")
    def api_metrics():
        """The full provenance report, or an explicit 'not measured' state."""
        rep = _load()
        if rep is None:
            return jsonify(
                available=False,
                reason="outputs/eval_report.json not found. Run: python -m src.evaluate",
            ), 200
        rep["available"] = True
        return jsonify(rep)

    @app.route("/api/training-history")
    def api_training_history():
        """Per-epoch train/val loss and accuracy, written by src.train."""
        path = os.path.join(outputs_dir, "training_histories.json")
        if not os.path.exists(path):
            return jsonify(
                available=False,
                reason="outputs/training_histories.json not found. Run: python -m src.train",
            ), 200
        try:
            with open(path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except (OSError, json.JSONDecodeError):
            return jsonify(available=False, reason="Could not read training history."), 200

    @app.route("/api/stats")
    def api_stats():
        """Dataset counts and headline test metrics, read from the report."""
        rep = _load()
        if rep is None:
            return jsonify(
                available=False,
                reason="No evaluation report yet. Run: python -m src.evaluate",
            ), 200

        best = next((m for m in rep["models"] if m["name"] == rep["best"]), rep["models"][0])
        return jsonify(
            available=True,
            generated_at=rep["generated_at"],
            num_classes=rep["num_classes"],
            classes=rep["classes"],
            dataset=rep.get("dataset"),
            evaluation=rep.get("evaluation"),
            best_model={
                "name": best["name"],
                "accuracy": best["accuracy"],
                "f1": best["f1"],
                "params": best["params"],
                "size_mb": best["size_mb"],
                "latency_median_ms": best["latency"]["median_ms"],
            },
        )

    return app
