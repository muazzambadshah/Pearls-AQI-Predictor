"""
Feature engineering for *direct multi-horizon* AQI forecasting.

The problem framing
-------------------
We want AQI up to 72 hours ahead. There are two ways to do that:

  1. Recursive - train a 1-step model, feed its own output back in, repeat 72
     times. Errors compound, and every exogenous input has to be guessed.
  2. Direct - train a model that maps (state at time `t`, horizon `h`) straight
     to `aqi[t + h]`.

This module implements (2). One model is trained across every horizon with `h`
itself as an input feature, so a single fit covers the whole 1-72h range while
each prediction is made in one shot rather than 72 chained shots.

The leakage discipline
----------------------
Every column produced here falls into exactly one of three buckets:

  ORIGIN   - computed from observations at or before the forecast origin `t`.
             Lags, rolling statistics and origin-time deltas all live here.
             Safe: at prediction time we genuinely know all of this.

  FUTURE   - weather and calendar values at the *target* time `t + h`.
             Safe: numerical weather prediction genuinely provides these ahead
             of time, and a calendar is known forever in advance.

  HORIZON  - the lead time itself.

What is deliberately absent: any pollutant or AQI measurement at `t + h`. Those
are exactly what we are trying to predict, and Open-Meteo's `us_aqi` is derived
from the pollutant concentrations, so including them at target time would hand
the model the answer. `tests/test_leakage.py` proves the separation holds by
perturbing the target column and asserting the feature matrix is bit-identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------
# Lags (hours before the origin) applied to the AQI history.
AQI_LAGS = (1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72)
POLLUTANT_LAGS = (1, 3, 6, 12, 24, 48)

# Rolling windows (hours, ending at and including the origin).
ROLLING_WINDOWS = (3, 6, 12, 24, 72, 168)

# Origin-time differences: aqi[t] - aqi[t - k]. Strictly backward looking.
DELTA_SPANS = (1, 3, 6, 12, 24, 48)

# Pollutants carried as origin-time state.
STATE_POLLUTANTS = ("pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone")

# Weather variables read at the target hour (known ahead from the forecast).
FUTURE_WEATHER = ("temperature_2m", "relative_humidity_2m", "dew_point_2m",
                  "apparent_temperature", "precipitation", "surface_pressure",
                  "cloud_cover", "wind_speed_10m", "wind_direction_10m",
                  "wind_gusts_10m", "wind_speed_100m", "wind_direction_100m",
                  "vapour_pressure_deficit")

# Rolling windows applied to the *future* weather path (also legitimately known).
FUTURE_WEATHER_WINDOWS = (6, 24)

ANCHOR_COLUMN = "aqi_at_origin"

# Bookkeeping columns that are never model inputs.
META_COLUMNS = ("origin", "target_time", "y")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _wind_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose wind speed/direction into u/v components.

    Direction in degrees is circular - 359 and 1 are adjacent but numerically far
    apart, which trees and linear models both handle badly. u/v components are
    continuous and carry the same information.
    """
    out = pd.DataFrame(index=df.index)
    for level in ("10m", "100m"):
        speed_col, dir_col = f"wind_speed_{level}", f"wind_direction_{level}"
        if speed_col in df.columns and dir_col in df.columns:
            rad = np.deg2rad(df[dir_col].astype(float))
            speed = df[speed_col].astype(float)
            out[f"wind_u_{level}"] = speed * np.cos(rad)
            out[f"wind_v_{level}"] = speed * np.sin(rad)
    return out


def _dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxies for how readily the atmosphere disperses what is emitted into it.

    Boundary-layer height would be the textbook variable here, but Open-Meteo's
    archive does not carry it (see `data_sources.WEATHER_VARS`), so these are
    built from the 10m/100m wind pair, which is available throughout:

      wind_shear        100m minus 10m wind speed. Strong shear means the
                        surface layer is coupled to faster air above and
                        pollutants mix upward; near-zero shear means a decoupled,
                        stagnant surface layer - the classic smog setup.
      ventilation_proxy geometric mean of the two wind speeds, standing in for
                        the wind x mixing-height product.
      stagnation        the inverse - large when both levels are calm, which is
                        when concentrations climb fastest.
    """
    out = pd.DataFrame(index=df.index)
    has_10m = "wind_speed_10m" in df.columns
    has_100m = "wind_speed_100m" in df.columns

    if has_10m and has_100m:
        w10 = df["wind_speed_10m"].astype(float)
        w100 = df["wind_speed_100m"].astype(float)
        out["wind_shear"] = w100 - w10
        out["ventilation_proxy"] = np.sqrt(np.clip(w10 * w100, 0, None))
        out["stagnation_index"] = 1.0 / (1.0 + out["ventilation_proxy"])
    elif has_10m:
        w10 = df["wind_speed_10m"].astype(float)
        out["ventilation_proxy"] = w10
        out["stagnation_index"] = 1.0 / (1.0 + w10)

    return out


def _calendar_features(index: pd.DatetimeIndex, prefix: str = "") -> pd.DataFrame:
    """Cyclical calendar encodings for a set of timestamps."""
    out = pd.DataFrame(index=index)
    hour = index.hour.values.astype(float)
    month = index.month.values.astype(float)
    dow = index.dayofweek.values.astype(float)
    doy = index.dayofyear.values.astype(float)

    out[f"{prefix}hour"] = hour
    out[f"{prefix}day"] = index.day.values.astype(float)
    out[f"{prefix}month"] = month
    out[f"{prefix}day_of_week"] = dow
    out[f"{prefix}is_weekend"] = (dow >= 5).astype(float)
    out[f"{prefix}hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out[f"{prefix}hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out[f"{prefix}month_sin"] = np.sin(2 * np.pi * month / 12)
    out[f"{prefix}month_cos"] = np.cos(2 * np.pi * month / 12)
    out[f"{prefix}doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out[f"{prefix}doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


# ---------------------------------------------------------------------------
# ORIGIN features - everything knowable at time t
# ---------------------------------------------------------------------------
def build_origin_features(raw: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    Features describing the atmospheric state at the forecast origin.

    Indexed by origin timestamp `t`. Every column is a function of observations
    at or before `t` only, so the frame stays valid no matter which horizon it
    is later paired with.
    """
    target_col = target_col or config.TARGET_COLUMN
    raw = raw.sort_index()

    # Columns are accumulated in a dict and assembled once. Assigning ~150
    # columns onto a DataFrame one at a time re-blocks it on every insert, which
    # is both slow and noisy with fragmentation warnings at this width.
    cols: dict[str, pd.Series] = {}
    target = raw[target_col].astype(float)

    # Persistence anchor: the most recent observed AQI. The single most
    # informative feature at short horizons, and the baseline every model must beat.
    cols[ANCHOR_COLUMN] = target

    for lag in AQI_LAGS:
        cols[f"aqi_lag_{lag}h"] = target.shift(lag)

    for window in ROLLING_WINDOWS:
        roll = target.rolling(window, min_periods=max(2, window // 4))
        cols[f"aqi_rollmean_{window}h"] = roll.mean()
        cols[f"aqi_rollstd_{window}h"] = roll.std()
        cols[f"aqi_rollmin_{window}h"] = roll.min()
        cols[f"aqi_rollmax_{window}h"] = roll.max()

    # Origin-time momentum: aqi[t] - aqi[t-k], strictly backward looking.
    for span in DELTA_SPANS:
        cols[f"aqi_delta_{span}h"] = target - target.shift(span)

    # Where the current level sits inside its recent range - a normalised
    # "unusually high for this week" signal that transfers across seasons.
    roll_24 = target.rolling(24, min_periods=6)
    cols["aqi_zscore_24h"] = (target - roll_24.mean()) / roll_24.std().replace(0, np.nan)

    # Pollutant state and momentum at the origin.
    for col in STATE_POLLUTANTS:
        if col not in raw.columns:
            continue
        series = raw[col].astype(float)
        cols[f"{col}_at_origin"] = series
        for lag in POLLUTANT_LAGS:
            cols[f"{col}_lag_{lag}h"] = series.shift(lag)
        cols[f"{col}_rollmean_24h"] = series.rolling(24, min_periods=6).mean()
        cols[f"{col}_delta_24h"] = series - series.shift(24)

    if {"pm2_5", "pm10"}.issubset(raw.columns):
        # Fine/coarse split separates combustion haze from dust events.
        cols["pm_ratio_at_origin"] = (
            raw["pm2_5"].astype(float) / raw["pm10"].astype(float).replace(0, np.nan)
        )

    # Weather state at the origin.
    for col in FUTURE_WEATHER:
        if col in raw.columns:
            cols[f"{col}_at_origin"] = raw[col].astype(float)

    for col, series in _wind_vector(raw).items():
        cols[f"{col}_at_origin"] = series
    for col, series in _dispersion(raw).items():
        cols[f"{col}_at_origin"] = series

    if "precipitation" in raw.columns:
        cols["precip_sum_24h_at_origin"] = (
            raw["precipitation"].astype(float).rolling(24, min_periods=6).sum()
        )

    feats = pd.DataFrame(cols, index=raw.index)
    feats.index.name = "origin"
    return feats


# ---------------------------------------------------------------------------
# FUTURE features - everything knowable about the target hour in advance
# ---------------------------------------------------------------------------
def build_future_features(weather_path: pd.DataFrame) -> pd.DataFrame:
    """
    Features describing the target hour `t + h`.

    `weather_path` must contain only forecastable weather columns - never
    pollutants or AQI. Indexed by target timestamp.
    """
    weather_path = weather_path.sort_index()
    out = _calendar_features(weather_path.index, prefix="tgt_")

    for col in FUTURE_WEATHER:
        if col in weather_path.columns:
            out[f"tgt_{col}"] = weather_path[col].astype(float)

    for col, series in _wind_vector(weather_path).items():
        out[f"tgt_{col}"] = series
    for col, series in _dispersion(weather_path).items():
        out[f"tgt_{col}"] = series

    # Smoothed weather leading into the target hour. A forecast covers the whole
    # path, so trailing windows here are legitimately known in advance.
    for window in FUTURE_WEATHER_WINDOWS:
        for col in ("wind_speed_10m", "boundary_layer_height", "temperature_2m",
                    "relative_humidity_2m"):
            if col in weather_path.columns:
                out[f"tgt_{col}_rollmean_{window}h"] = (
                    weather_path[col].astype(float).rolling(window, min_periods=1).mean()
                )
        if "precipitation" in weather_path.columns:
            out[f"tgt_precip_sum_{window}h"] = (
                weather_path["precipitation"].astype(float)
                .rolling(window, min_periods=1).sum()
            )

    out.index.name = "target_time"
    return out


# ---------------------------------------------------------------------------
# Supervised dataset assembly
# ---------------------------------------------------------------------------
def make_supervised(raw: pd.DataFrame,
                    horizons=None,
                    target_col: str | None = None,
                    origin_stride: int = 1,
                    dropna_features: bool = True) -> pd.DataFrame:
    """
    Build the flat supervised table used for training and evaluation.

    One row per (origin `t`, horizon `h`) pair, carrying:
        origin features at `t`  +  future features at `t + h`  +  `h`  +  y

    Parameters
    ----------
    origin_stride
        Keep every Nth origin. Horizons multiply the row count by up to 72, so a
        stride is the cheapest way to hold memory in range without dropping any
        horizon from the training distribution.
    """
    target_col = target_col or config.TARGET_COLUMN
    horizons = tuple(horizons or config.TRAIN_HORIZONS)
    raw = raw.sort_index()

    origin_feats = build_origin_features(raw, target_col=target_col)

    # Future features come from the observed weather path. During training the
    # observed weather doubles as a "perfect forecast"; at inference the same
    # columns are filled from the real Open-Meteo forecast. This is the standard
    # perfect-prog setup, and the report quantifies what it costs.
    weather_cols = [c for c in FUTURE_WEATHER if c in raw.columns]
    future_feats = build_future_features(raw[weather_cols])

    target = raw[target_col].astype(float)

    if origin_stride > 1:
        origin_feats = origin_feats.iloc[::origin_stride]

    # Drop origins whose history is too short to have real lag values.
    if dropna_features:
        origin_feats = origin_feats.dropna()
    if origin_feats.empty:
        return pd.DataFrame()

    blocks = []
    for h in horizons:
        target_times = origin_feats.index + pd.Timedelta(hours=int(h))

        y = target.reindex(target_times)
        fut = future_feats.reindex(target_times)

        valid = y.notna().values & fut.notna().all(axis=1).values
        if not valid.any():
            continue

        block = origin_feats.loc[valid].reset_index(drop=True)
        block = pd.concat([block, fut.loc[valid].reset_index(drop=True)], axis=1)

        # Downcast per block rather than after the concat. At 72 horizons the
        # combined frame is the memory high-water mark of the whole pipeline, and
        # float32 halves it - the inputs are weather readings with three or four
        # significant figures, so the extra precision buys nothing.
        block = block.astype("float32", copy=False)

        block["horizon_h"] = np.float32(h)
        block["horizon_days"] = np.float32(h / 24.0)
        block["origin"] = origin_feats.index[valid]
        block["target_time"] = target_times[valid]
        block["y"] = y.values[valid].astype("float32")
        blocks.append(block)

    if not blocks:
        return pd.DataFrame()

    dataset = pd.concat(blocks, ignore_index=True)
    del blocks

    # Order by (origin, horizon) so any consumer that assumes chronological rows
    # - the deep model's validation tail, for one - sees them that way.
    # `sort_values(...).reset_index()` copies the frame twice; lexsort plus a
    # single `take` copies it once, which matters at this width.
    order = np.lexsort((dataset["horizon_h"].to_numpy(),
                        dataset["origin"].to_numpy()))
    return dataset.take(order).reset_index(drop=True)


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Model input columns - everything except the bookkeeping fields."""
    return [c for c in dataset.columns if c not in META_COLUMNS]


def split_feature_matrix(dataset: pd.DataFrame):
    """Return `(X, y, anchor)` ready for a scikit-learn estimator."""
    cols = feature_columns(dataset)
    X = dataset[cols].astype("float32")
    y = dataset["y"].astype("float64")
    anchor = dataset[ANCHOR_COLUMN].astype("float64")
    return X, y, anchor


# ---------------------------------------------------------------------------
# Inference-time assembly
# ---------------------------------------------------------------------------
def build_inference_frame(observed: pd.DataFrame,
                          future_weather: pd.DataFrame,
                          horizons=None,
                          target_col: str | None = None) -> pd.DataFrame:
    """
    Build the feature rows for a live forecast.

    `observed` is history up to the forecast origin; `future_weather` is the real
    weather forecast for the hours after it. Produces one row per horizon through
    exactly the same code paths as training, so train/serve skew cannot creep in.
    """
    target_col = target_col or config.TARGET_COLUMN
    horizons = tuple(horizons or config.TRAIN_HORIZONS)

    origin_feats = build_origin_features(observed, target_col=target_col).dropna()
    if origin_feats.empty:
        raise ValueError(
            "Not enough observation history to compute origin features - "
            "at least 7 days of continuous hourly data is required."
        )
    origin_row = origin_feats.iloc[[-1]]
    origin_ts = origin_row.index[0]

    weather_cols = [c for c in FUTURE_WEATHER if c in future_weather.columns]
    if not weather_cols:
        raise ValueError("future_weather contains no usable weather columns")

    # Prepend recent observed weather so trailing rolling windows over the future
    # path start from real history rather than a cold start.
    hist_weather = observed[[c for c in weather_cols if c in observed.columns]].tail(48)
    weather_path = pd.concat([hist_weather, future_weather[weather_cols]])
    weather_path = weather_path[~weather_path.index.duplicated(keep="last")].sort_index()

    future_feats = build_future_features(weather_path)

    rows = []
    for h in horizons:
        target_time = origin_ts + pd.Timedelta(hours=int(h))
        if target_time not in future_feats.index:
            continue
        row = origin_row.reset_index(drop=True).copy()
        fut = future_feats.loc[[target_time]].reset_index(drop=True)
        row = pd.concat([row, fut], axis=1)
        row["horizon_h"] = float(h)
        row["horizon_days"] = float(h) / 24.0
        row["origin"] = origin_ts
        row["target_time"] = target_time
        rows.append(row)

    if not rows:
        raise ValueError("Weather forecast does not cover any requested horizon")

    return pd.concat(rows, ignore_index=True)
