"""
Central configuration for the Pearls AQI Predictor.

Every value can be overridden through an environment variable (see .env.example),
so the same code runs unchanged on a laptop, in GitHub Actions, or in a container.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional convenience - load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is not required
    pass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------
CITY_NAME = os.getenv("CITY_NAME", "Lahore")
LATITUDE = _env_float("CITY_LAT", 31.5497)
LONGITUDE = _env_float("CITY_LON", 74.3436)
TIMEZONE = os.getenv("TIMEZONE", "Asia/Karachi")

# ---------------------------------------------------------------------------
# Forecast problem definition
# ---------------------------------------------------------------------------
FORECAST_HORIZON_DAYS = _env_int("FORECAST_HORIZON_DAYS", 3)
FORECAST_HORIZON_HOURS = FORECAST_HORIZON_DAYS * 24  # 72h

# Horizons the model is trained on. We train ONE model across all horizons with
# the horizon itself as a feature (a "direct multi-horizon" formulation), which
# avoids the error compounding of recursive one-step forecasting.
TRAIN_HORIZONS = tuple(range(1, FORECAST_HORIZON_HOURS + 1))

TARGET_COLUMN = "aqi"

# ---------------------------------------------------------------------------
# Data source
#   open_meteo -> free, no API key, real historical archive + real forecasts
#   aqicn      -> https://aqicn.org/api/ (free token, current reading only)
# ---------------------------------------------------------------------------
DATA_SOURCE = os.getenv("DATA_SOURCE", "open_meteo")
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")

# Open-Meteo's air-quality archive only carries a populated `us_aqi` field from
# 2023-01-01 onwards (verified empirically), so that is our earliest usable date.
HISTORY_START_DATE = os.getenv("HISTORY_START_DATE", "2023-01-01")

# ---------------------------------------------------------------------------
# Feature store backend: "local" (parquet, zero setup) or "hopsworks"
# ---------------------------------------------------------------------------
FEATURE_STORE_BACKEND = os.getenv("FEATURE_STORE_BACKEND", "local")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", BASE_DIR / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw hourly observations are the single source of truth. Engineered features
# are derived from them on demand, so feature logic can change without a refetch.
RAW_OBSERVATIONS_PATH = DATA_DIR / "raw_observations.parquet"
PREDICTIONS_PATH = DATA_DIR / "predictions.parquet"
BACKTEST_PATH = DATA_DIR / "backtest_results.parquet"
HORIZON_METRICS_PATH = DATA_DIR / "horizon_metrics.parquet"

MODEL_REGISTRY_PATH = MODELS_DIR / "registry.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training controls
# ---------------------------------------------------------------------------
# Fraction of the timeline held out (chronologically) as the final test set.
TEST_FRACTION = _env_float("TEST_FRACTION", 0.2)
# Number of expanding-window folds used by the walk-forward backtest.
N_BACKTEST_FOLDS = _env_int("N_BACKTEST_FOLDS", 4)
# Cap on rows in the horizon-expanded dataset. Sized for a 16GB machine: at
# ~160 float32 features this is roughly 160MB for the frame, leaving headroom
# for the ensembles to train alongside it.
MAX_TRAIN_ROWS = _env_int("MAX_TRAIN_ROWS", 250_000)
RANDOM_SEED = _env_int("RANDOM_SEED", 42)

# Enable the (heavier) deep-learning candidates. Off by default so CI stays fast.
ENABLE_DEEP_MODEL = os.getenv("ENABLE_DEEP_MODEL", "0") == "1"   # PyTorch MLP
ENABLE_TF_MODEL = os.getenv("ENABLE_TF_MODEL", "0") == "1"       # TensorFlow/Keras MLP

# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
ALERT_AQI_THRESHOLD = _env_float("ALERT_AQI_THRESHOLD", 150.0)
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
