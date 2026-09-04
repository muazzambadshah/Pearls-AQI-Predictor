"""
API contract tests.

Run against the real FastAPI app with a TestClient. The endpoints that need a
trained model skip cleanly when the pipeline has not been run, so the suite
passes on a fresh checkout while still verifying the full contract once a model
exists.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import config, model_registry
from app.api import app

client = TestClient(app)


def _pipeline_ready() -> bool:
    try:
        return (config.BEST_MODEL_PATH.exists()
                and model_registry.production_entry() is not None)
    except Exception:  # noqa: BLE001
        return False


needs_model = pytest.mark.skipif(
    not _pipeline_ready(),
    reason="no trained model - run `python -m src.backfill && python -m src.train`",
)


def test_root_lists_endpoints():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["city"] == config.CITY_NAME
    for path in ("/health", "/forecast", "/alerts"):
        assert path in body["endpoints"]


def test_health_always_answers():
    """A readiness probe must get a response even when nothing is set up."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["model_ready"], bool)
    assert isinstance(body["data_ready"], bool)


@needs_model
def test_health_reports_ready_when_pipeline_is_complete():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_ready"] and body["data_ready"]
    assert body["observations"] > 1000


@needs_model
def test_current_conditions_shape():
    body = client.get("/current").json()
    assert 0 <= body["aqi"] <= 500
    assert body["category"]
    assert body["advice"]
    assert isinstance(body["pollutants"], dict)


@needs_model
def test_forecast_returns_requested_horizon():
    body = client.get("/forecast", params={"hours": 24}).json()
    assert body["horizon_hours"] == 24
    assert len(body["points"]) == 24

    for point in body["points"]:
        assert 0 <= point["predicted_aqi"] <= 500
        assert point["category"]
        assert point["color"].startswith("#")

    horizons = [p["horizon_h"] for p in body["points"]]
    assert horizons == sorted(horizons)


@needs_model
def test_forecast_horizon_bounds_are_enforced():
    assert client.get("/forecast", params={"hours": 0}).status_code == 422
    assert client.get("/forecast", params={"hours": 500}).status_code == 422


@needs_model
def test_uncertainty_band_brackets_the_point_forecast():
    points = client.get("/forecast", params={"hours": 48}).json()["points"]
    banded = [p for p in points if p["lower_80"] is not None]
    assert banded, "expected uncertainty bands once a backtest has run"

    for p in banded:
        assert p["lower_80"] <= p["predicted_aqi"] <= p["upper_80"]

    # Bands must widen with lead time - that is the point of them.
    first = next(p for p in banded if p["horizon_h"] <= 3)
    last = max(banded, key=lambda p: p["horizon_h"])
    assert (last["upper_80"] - last["lower_80"]) > (first["upper_80"] - first["lower_80"])


@needs_model
def test_daily_forecast_returns_three_days():
    body = client.get("/forecast/daily").json()
    assert 3 <= len(body) <= 4
    for day in body:
        assert day["min_aqi"] <= day["avg_aqi"] <= day["max_aqi"]
        assert day["advice"]


@needs_model
def test_alerts_respect_the_threshold():
    high = client.get("/alerts", params={"threshold": 500}).json()
    assert high["alert"] is False

    low = client.get("/alerts", params={"threshold": 1}).json()
    assert low["alert"] is True
    assert low["episodes"]


@needs_model
def test_model_endpoint_exposes_the_production_card():
    body = client.get("/model").json()
    assert body["production"]["model_name"]
    assert "rmse" in body["production"]["metrics"]
    assert isinstance(body["leaderboard"], list)


@needs_model
def test_metrics_endpoint_returns_the_horizon_curve():
    resp = client.get("/metrics")
    if resp.status_code == 503:
        pytest.skip("no backtest metrics recorded yet")

    rows = resp.json()["by_horizon"]
    assert len(rows) > 10
    assert all("rmse" in r for r in rows)

    # Error must grow with lead time; compare the ends rather than demanding
    # strict monotonicity, which sampling noise would break.
    early = [r["rmse"] for r in rows if r["horizon_h"] <= 6]
    late = [r["rmse"] for r in rows if r["horizon_h"] >= 60]
    if early and late:
        assert sum(late) / len(late) > sum(early) / len(early)
