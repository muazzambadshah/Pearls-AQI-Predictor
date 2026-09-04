"""
FastAPI service.

The programmatic face of the project: the dashboard consumes it, and so can
anything else - a mobile client, a monitoring probe, a webhook subscriber.

Forecasts are cached in-process with a short TTL. Generating one costs two
upstream API calls plus a model pass, and the underlying data only refreshes
hourly, so recomputing per request would be pure waste and would hammer
Open-Meteo for no new information.

Run with:
    uvicorn app.api:app --reload --port 8000

Interactive docs at /docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import alerts, config, feature_store, model_registry, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def _json_safe(value):
    """
    Recursively replace NaN/Inf with None.

    Registry entries are plain JSON on disk but pass through pandas/numpy on
    the way there, so a NaN can legitimately end up in a metrics dict (e.g. a
    blend model's train_rows before the aggregate_folds fix, or a std-dev
    computed over a single fold). The standard library's json encoder treats
    NaN as a hard error rather than coercing it, and would otherwise take
    down any endpoint that echoes a metrics dict verbatim.
    """
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Forecast cache
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS = 900  # 15 minutes

_cache: dict = {
    "forecast": None,
    "fetched_at": 0.0,
}


def _load_disk_forecast() -> pd.DataFrame | None:
    """
    Try to load the most recent forecast saved by predict.forecast().

    This is useful when Open-Meteo is temporarily unavailable.
    """

    try:
        frame = predict.load_cached_forecast()

        if frame is not None and not frame.empty:
            logger.info(
                "Loaded cached forecast from %s",
                config.PREDICTIONS_PATH,
            )

            return frame

    except Exception as exc:
        logger.warning(
            "Could not load cached forecast: %s",
            exc,
        )

    return None


def _cached_forecast(
    force: bool = False,
) -> pd.DataFrame:
    """
    Return a forecast using the following strategy:

    1. Use in-memory cache if still fresh.
    2. Try to generate a fresh forecast.
    3. If upstream APIs fail, use the most recent disk forecast.
    4. If nothing exists, return a proper 503 error.
    """

    now = time.time()

    cached = _cache.get("forecast")
    fetched_at = _cache.get("fetched_at", 0.0)

    fresh = (
        cached is not None
        and not cached.empty
        and (now - fetched_at) < CACHE_TTL_SECONDS
    )

    if fresh and not force:
        logger.info(
            "Serving forecast from in-memory cache"
        )
        return cached

    # ------------------------------------------------------------------
    # Try fresh forecast
    # ------------------------------------------------------------------

    try:
        frame = predict.forecast()

        if frame is None or frame.empty:
            raise ValueError(
                "Forecast generation returned no data."
            )

        _cache["forecast"] = frame
        _cache["fetched_at"] = now

        logger.info(
            "Fresh forecast generated successfully"
        )

        return frame

    except FileNotFoundError as exc:

        logger.warning(
            "Model/data not ready: %s",
            exc,
        )

    except Exception as exc:

        logger.warning(
            "Fresh forecast generation failed: %s",
            exc,
        )

    # ------------------------------------------------------------------
    # Fresh forecast failed.
    #
    # Try the in-memory cache even if its TTL has expired.
    # ------------------------------------------------------------------

    if cached is not None and not cached.empty:

        logger.warning(
            "Serving stale in-memory forecast because "
            "fresh forecast generation failed."
        )

        return cached

    # ------------------------------------------------------------------
    # Try the forecast saved on disk.
    # ------------------------------------------------------------------

    disk_forecast = _load_disk_forecast()

    if disk_forecast is not None:

        _cache["forecast"] = disk_forecast
        _cache["fetched_at"] = now

        logger.warning(
            "Serving disk-cached forecast because "
            "fresh forecast generation failed."
        )

        return disk_forecast

    # ------------------------------------------------------------------
    # Nothing available.
    # ------------------------------------------------------------------

    raise HTTPException(
        status_code=503,
        detail=(
            "Forecast is currently unavailable. "
            "Open-Meteo could not be reached and no cached "
            "forecast is available."
        ),
    )


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):

    logger.info(
        "AQI API starting for %s",
        config.CITY_NAME,
    )

    yield

    logger.info(
        "AQI API shutting down"
    )


app = FastAPI(
    title="Pearls AQI Predictor API",
    description=(
        "Three-day hourly Air Quality Index forecasts from a direct "
        "multi-horizon model, with SHAP explanations and hazardous-air alerts."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    timestamp: str
    predicted_aqi: float
    horizon_h: int
    category: str
    color: str
    lower_80: float | None = None
    upper_80: float | None = None


class ForecastResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    forecast_origin: str
    generated_at: str
    model_name: str
    horizon_hours: int
    points: list[ForecastPoint]


class DaySummary(BaseModel):
    date: str
    min_aqi: float
    avg_aqi: float
    max_aqi: float
    peak_hour: str
    category: str
    color: str
    advice: str


class HealthResponse(BaseModel):
    status: str
    city: str
    model_ready: bool
    data_ready: bool
    model_name: str | None = None
    observations: int = 0
    latest_observation: str | None = None
    detail: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root() -> dict:

    return {
        "service": "Pearls AQI Predictor",
        "city": config.CITY_NAME,
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/current",
            "/forecast",
            "/forecast/daily",
            "/alerts",
            "/explain",
            "/model",
            "/metrics",
        ],
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
)
def health() -> HealthResponse:
    """
    Liveness and readiness.

    Never raises - monitoring systems need a response instead of a 500.
    """

    store_info = {}
    data_ready = False
    observations = 0
    latest = None

    try:

        store_info = feature_store.describe()

        data_ready = bool(
            store_info.get("available")
        )

        observations = int(
            store_info.get("rows", 0)
        )

        latest = store_info.get("end")

    except Exception as exc:

        logger.warning(
            "Feature store check failed: %s",
            exc,
        )

    entry = None

    try:

        entry = (
            model_registry.production_entry()
        )

    except Exception as exc:

        logger.warning(
            "Registry check failed: %s",
            exc,
        )

    model_ready = (
        entry is not None
        and config.BEST_MODEL_PATH.exists()
    )

    ready = (
        model_ready
        and data_ready
    )

    return HealthResponse(
        status="ok" if ready else "degraded",
        city=config.CITY_NAME,
        model_ready=model_ready,
        data_ready=data_ready,
        model_name=(
            entry.get("model_name")
            if entry
            else None
        ),
        observations=observations,
        latest_observation=latest,
        detail=(
            None
            if ready
            else (
                "Run `python -m src.backfill` "
                "then `python -m src.train`."
            )
        ),
    )


# ---------------------------------------------------------------------------
# Current conditions
# ---------------------------------------------------------------------------

@app.get(
    "/current",
    tags=["forecast"],
)
def current() -> dict:
    """
    Latest observed AQI, category, and pollutant breakdown.
    """

    try:

        return predict.current_conditions()

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Current conditions failed"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                f"Current conditions failed: {exc}"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

@app.get(
    "/forecast",
    response_model=ForecastResponse,
    tags=["forecast"],
)
def forecast(
    hours: int = Query(
        default=config.FORECAST_HORIZON_HOURS,
        ge=1,
        le=120,
    ),
    refresh: bool = Query(
        default=False,
        description="Bypass cache and recompute.",
    ),
) -> ForecastResponse:
    """
    Hourly AQI forecast with 80% uncertainty bands.
    """

    frame = _cached_forecast(
        force=refresh
    ).head(hours)

    if frame.empty:

        raise HTTPException(
            status_code=503,
            detail="Forecast returned no points.",
        )

    entry = (
        model_registry.production_entry()
        or {}
    )

    points = []

    for ts, row in frame.iterrows():

        lower = row.get("lower_80")
        upper = row.get("upper_80")

        if pd.isna(lower):
            lower_value = None
        else:
            lower_value = round(
                float(lower),
                1,
            )

        if pd.isna(upper):
            upper_value = None
        else:
            upper_value = round(
                float(upper),
                1,
            )

        points.append(
            ForecastPoint(
                timestamp=str(ts),
                predicted_aqi=round(
                    float(
                        row["predicted_aqi"]
                    ),
                    1,
                ),
                horizon_h=int(
                    row["horizon_h"]
                ),
                category=str(
                    row["category"]
                ),
                color=str(
                    row["color"]
                ),
                lower_80=lower_value,
                upper_80=upper_value,
            )
        )

    return ForecastResponse(
        city=config.CITY_NAME,
        latitude=config.LATITUDE,
        longitude=config.LONGITUDE,
        forecast_origin=str(
            frame["forecast_origin"].iloc[0]
        ),
        generated_at=(
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        ),
        model_name=entry.get(
            "model_name",
            "unknown",
        ),
        horizon_hours=len(points),
        points=points,
    )


# ---------------------------------------------------------------------------
# Daily forecast
# ---------------------------------------------------------------------------

@app.get(
    "/forecast/daily",
    response_model=list[DaySummary],
    tags=["forecast"],
)
def forecast_daily() -> list[DaySummary]:
    """
    Day-by-day rollup:
    min/mean/max and health advice.
    """

    try:

        frame = _cached_forecast()

        if frame.empty:

            raise HTTPException(
                status_code=503,
                detail="Forecast returned no data.",
            )

        summary = predict.daily_summary(
            frame
        )

        result = []

        for _, row in summary.iterrows():

            result.append(
                DaySummary(
                    date=str(
                        row["date"]
                    ),
                    min_aqi=round(
                        float(
                            row["min_aqi"]
                        ),
                        1,
                    ),
                    avg_aqi=round(
                        float(
                            row["avg_aqi"]
                        ),
                        1,
                    ),
                    max_aqi=round(
                        float(
                            row["max_aqi"]
                        ),
                        1,
                    ),
                    peak_hour=str(
                        row["peak_hour"]
                    ),
                    category=str(
                        row["category"]
                    ),
                    color=str(
                        row["color"]
                    ),
                    advice=str(
                        row["advice"]
                    ),
                )
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Daily forecast failed"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                f"Forecast failed: {exc}"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@app.get(
    "/alerts",
    tags=["alerts"],
)
def get_alerts(
    threshold: float = Query(
        default=config.ALERT_AQI_THRESHOLD,
        ge=0,
        le=500,
    ),
) -> dict:
    """
    Hazardous-air episodes in the forecast window.
    """

    try:

        frame = _cached_forecast()

        result = alerts.build_alert(
            frame,
            threshold=threshold,
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Alert generation failed"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                f"Alert generation failed: {exc}"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

@app.get(
    "/explain",
    tags=["analytics"],
)
def explain(
    top_n: int = Query(
        default=15,
        ge=1,
        le=60,
    ),
) -> dict:
    """
    SHAP feature importance.
    """

    from src import explainability

    try:

        importance, groups = (
            explainability
            .explain_production_model()
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Explanation failed: {exc}"
            ),
        ) from exc

    return {
        "features": (
            importance
            .head(top_n)
            .to_dict("records")
        ),
        "groups": (
            groups
            .to_dict("records")
        ),
    }


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------

@app.get(
    "/model",
    tags=["analytics"],
)
def model_info() -> dict:
    """
    Production model card plus leaderboard.
    """

    entry = (
        model_registry.production_entry()
    )

    if entry is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "No model has been promoted yet"
            ),
        )

    return _json_safe({
        "production": {
            "model_name": entry.get(
                "model_name"
            ),
            "trained_at": entry.get(
                "trained_at"
            ),
            "promoted_at": entry.get(
                "promoted_at"
            ),
            "metrics": entry.get(
                "metrics"
            ),
            "n_features": entry.get(
                "n_features"
            ),
            "data_fingerprint": entry.get(
                "data_fingerprint"
            ),
            "selection": entry.get(
                "selection"
            ),
        },
        "leaderboard": [
            {
                "model_name": e.get(
                    "model_name"
                ),
                "trained_at": e.get(
                    "trained_at"
                ),
                "rmse": e.get(
                    "metrics", {}
                ).get("rmse"),
                "mae": e.get(
                    "metrics", {}
                ).get("mae"),
                "r2": e.get(
                    "metrics", {}
                ).get("r2"),
            }
            for e in model_registry.leaderboard(
                limit=10
            )
        ],
    })


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@app.get(
    "/metrics",
    tags=["analytics"],
)
def metrics() -> dict:
    """
    Backtest accuracy by forecast lead time.
    """

    if not config.HORIZON_METRICS_PATH.exists():

        raise HTTPException(
            status_code=503,
            detail=(
                "No backtest metrics yet. "
                "Run `python -m src.train`."
            ),
        )

    frame = pd.read_parquet(
        config.HORIZON_METRICS_PATH
    )

    keep = [
        c
        for c in (
            "horizon_h",
            "rmse",
            "mae",
            "r2",
            "category_accuracy",
            "skill_vs_persistence",
            "persistence_rmse",
            "n",
        )
        if c in frame.columns
    ]

    return {
        "by_horizon": (
            frame[keep]
            .round(4)
            .to_dict("records")
        ),
        "feature_store": (
            feature_store.describe()
        ),
    }