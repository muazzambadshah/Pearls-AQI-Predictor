"""
Flask REST API - an alternate to app/api.py, exposing the identical routes and
JSON shapes so the same frontend works against either backend unmodified.

Two implementations exist because the brief names both frameworks. FastAPI is
the one actually served in production (async, typed response models, free
OpenAPI docs at /docs) - this Flask app is a straight, behaviour-identical
port for anyone who specifically needs the Flask/WSGI path, or a Flask-only
deployment target.

Run with (from the aqi_predictor/ directory):
    python -m app.flask_api
or:
    flask --app app.flask_api run --port 8000

Note: `python app/flask_api.py` (running the file directly rather than as a
module) will fail with `ModuleNotFoundError: No module named 'src'` - running
a script directly only puts its own folder (app/) on sys.path, not the repo
root where `src/` lives. The sys.path fix-up below makes the direct form work
too, so both commands behave the same.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

if __package__ in (None, ""):
    # Running as `python app/flask_api.py` rather than `python -m app.flask_api`
    # or via an installed package - put the repo root on sys.path so `src`
    # resolves either way.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import alerts, config, feature_store, model_registry, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 900  # 15 minutes; upstream data only moves hourly.
_cache: dict = {"forecast": None, "fetched_at": 0.0}


def _json_safe(value):
    """Recursively replace NaN/Inf with None - see app/api.py for why."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET"]}})


def _cached_forecast(force: bool = False) -> pd.DataFrame:
    now = time.time()
    fresh = _cache["forecast"] is not None and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS
    if fresh and not force:
        return _cache["forecast"]

    try:
        frame = predict.forecast()
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _cache["forecast"] is not None:
            logger.warning("Forecast refresh failed (%s); serving cached copy", exc)
            return _cache["forecast"]
        raise

    _cache["forecast"] = frame
    _cache["fetched_at"] = now
    return frame


# ---------------------------------------------------------------------------
# Routes - same paths, same JSON keys as app/api.py
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return jsonify({
        "service": "Pearls AQI Predictor",
        "city": config.CITY_NAME,
        "docs": "/docs (FastAPI only - see app/api.py)",
        "endpoints": ["/health", "/current", "/forecast", "/forecast/daily",
                      "/alerts", "/explain", "/model", "/metrics"],
    })


@app.get("/health")
def health():
    store_info, data_ready, observations, latest = {}, False, 0, None
    try:
        store_info = feature_store.describe()
        data_ready = bool(store_info.get("available"))
        observations = int(store_info.get("rows", 0))
        latest = store_info.get("end")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feature store check failed: %s", exc)

    entry = None
    try:
        entry = model_registry.production_entry()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Registry check failed: %s", exc)

    model_ready = entry is not None and config.BEST_MODEL_PATH.exists()
    ready = model_ready and data_ready

    return jsonify({
        "status": "ok" if ready else "degraded",
        "city": config.CITY_NAME,
        "model_ready": model_ready,
        "data_ready": data_ready,
        "model_name": entry.get("model_name") if entry else None,
        "observations": observations,
        "latest_observation": latest,
        "detail": None if ready else "Run `python -m src.backfill` then `python -m src.train`.",
    })


@app.get("/current")
def current():
    try:
        return jsonify(predict.current_conditions())
    except FileNotFoundError as exc:
        return jsonify({"detail": str(exc)}), 503


@app.get("/forecast")
def forecast():
    hours = request.args.get("hours", default=config.FORECAST_HORIZON_HOURS, type=int)
    refresh = request.args.get("refresh", default=False, type=lambda v: v.lower() == "true")
    try:
        frame = _cached_forecast(force=refresh).head(hours)
    except FileNotFoundError as exc:
        return jsonify({"detail": f"Model or data not ready: {exc}"}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"detail": f"Forecast failed: {exc}"}), 502

    entry = model_registry.production_entry() or {}
    points = [
        {
            "timestamp": str(ts),
            "predicted_aqi": round(float(row["predicted_aqi"]), 1),
            "horizon_h": int(row["horizon_h"]),
            "category": str(row["category"]),
            "color": str(row["color"]),
            "lower_80": None if pd.isna(row.get("lower_80")) else round(float(row["lower_80"]), 1),
            "upper_80": None if pd.isna(row.get("upper_80")) else round(float(row["upper_80"]), 1),
        }
        for ts, row in frame.iterrows()
    ]
    return jsonify({
        "city": config.CITY_NAME,
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "forecast_origin": str(frame["forecast_origin"].iloc[0]) if len(frame) else "",
        "generated_at": pd.Timestamp.now().isoformat(),
        "model_name": entry.get("model_name", "unknown"),
        "horizon_hours": len(points),
        "points": points,
    })


@app.get("/forecast/daily")
def forecast_daily():
    try:
        summary = predict.daily_summary(_cached_forecast())
    except FileNotFoundError as exc:
        return jsonify({"detail": str(exc)}), 503
    return jsonify([
        {
            "date": str(row["date"]),
            "min_aqi": round(float(row["min_aqi"]), 1),
            "avg_aqi": round(float(row["avg_aqi"]), 1),
            "max_aqi": round(float(row["max_aqi"]), 1),
            "peak_hour": str(row["peak_hour"]),
            "category": str(row["category"]),
            "color": str(row["color"]),
            "advice": str(row["advice"]),
        }
        for _, row in summary.iterrows()
    ])


@app.get("/alerts")
def get_alerts():
    threshold = request.args.get("threshold", default=config.ALERT_AQI_THRESHOLD, type=float)
    try:
        return jsonify(alerts.build_alert(_cached_forecast(), threshold=threshold))
    except FileNotFoundError as exc:
        return jsonify({"detail": str(exc)}), 503


@app.get("/explain")
def explain():
    from src import explainability

    top_n = request.args.get("top_n", default=15, type=int)
    try:
        importance, groups = explainability.explain_production_model()
    except FileNotFoundError as exc:
        return jsonify({"detail": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"detail": f"Explanation failed: {exc}"}), 500

    return jsonify({
        "features": importance.head(top_n).to_dict("records"),
        "groups": groups.to_dict("records"),
    })


@app.get("/model")
def model_info():
    entry = model_registry.production_entry()
    if entry is None:
        return jsonify({"detail": "No model has been promoted yet"}), 503
    return jsonify(_json_safe({
        "production": {
            "model_name": entry.get("model_name"),
            "trained_at": entry.get("trained_at"),
            "promoted_at": entry.get("promoted_at"),
            "metrics": entry.get("metrics"),
            "n_features": entry.get("n_features"),
            "data_fingerprint": entry.get("data_fingerprint"),
            "selection": entry.get("selection"),
        },
        "leaderboard": [
            {"model_name": e.get("model_name"), "trained_at": e.get("trained_at"),
             "rmse": e.get("metrics", {}).get("rmse"),
             "mae": e.get("metrics", {}).get("mae"),
             "r2": e.get("metrics", {}).get("r2")}
            for e in model_registry.leaderboard(limit=10)
        ],
    }))


@app.get("/metrics")
def metrics():
    if not config.HORIZON_METRICS_PATH.exists():
        return jsonify({"detail": "No backtest metrics yet. Run `python -m src.train`."}), 503
    frame = pd.read_parquet(config.HORIZON_METRICS_PATH)
    keep = [c for c in ("horizon_h", "rmse", "mae", "r2", "category_accuracy",
                        "skill_vs_persistence", "persistence_rmse", "n")
            if c in frame.columns]
    return jsonify({
        "by_horizon": frame[keep].round(4).to_dict("records"),
        "feature_store": feature_store.describe(),
    })


if __name__ == "__main__":
    logger.info("AQI Flask API starting for %s", config.CITY_NAME)
    app.run(host="127.0.0.1", port=8000, debug=False)
