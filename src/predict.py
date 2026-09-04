"""
Inference.

Loads the production model, pulls the latest observations plus the real weather
forecast, and produces a 72-hour AQI forecast in a single pass - one direct
prediction per horizon, no recursion, no error compounding.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import (
    config,
    data_sources,
    feature_engineering,
    feature_store,
    model_registry,
)

logger = logging.getLogger(__name__)


# US EPA AQI categories:
# (upper bound, label, colour, advice).
AQI_BANDS = [
    (
        50,
        "Good",
        "#22c55e",
        "Air quality is satisfactory. Outdoor activity carries little or no risk.",
    ),
    (
        100,
        "Moderate",
        "#eab308",
        "Acceptable overall. Unusually sensitive people should consider limiting "
        "prolonged outdoor exertion.",
    ),
    (
        150,
        "Unhealthy for Sensitive Groups",
        "#f97316",
        "Children, older adults, and people with heart or lung conditions should "
        "reduce prolonged outdoor exertion.",
    ),
    (
        200,
        "Unhealthy",
        "#ef4444",
        "Everyone may begin to feel effects. Sensitive groups should avoid outdoor "
        "exertion; consider an N95 mask outdoors.",
    ),
    (
        300,
        "Very Unhealthy",
        "#a855f7",
        "Health alert. Avoid outdoor activity; run an air purifier indoors and keep "
        "windows closed.",
    ),
    (
        501,
        "Hazardous",
        "#7f1d1d",
        "Health emergency. Everyone should remain indoors with windows sealed "
        "and air filtration running.",
    ),
]


def aqi_category(aqi: float):
    """Return `(label, colour)` for an AQI value."""
    for upper, label, colour, _ in AQI_BANDS:
        if aqi < upper:
            return label, colour

    return AQI_BANDS[-1][1], AQI_BANDS[-1][2]


def health_advice(aqi: float) -> str:
    """Return health advice for an AQI value."""
    for upper, _, _, advice in AQI_BANDS:
        if aqi < upper:
            return advice

    return AQI_BANDS[-1][3]


def _normalize_datetime_index(
    index: pd.Index,
    name: str = "timestamp",
) -> pd.DatetimeIndex:
    """
    Normalize any datetime-like index to timezone-aware UTC.

    This prevents pandas errors when comparing:
      - naive timestamps
      - UTC timestamps
      - timestamps with other timezone names
      - pandas datetime64[us, Etc/UTC]
    """
    dt_index = pd.DatetimeIndex(index)

    if dt_index.tz is None:
        dt_index = dt_index.tz_localize("UTC")
    else:
        dt_index = dt_index.tz_convert("UTC")

    dt_index.name = name

    return dt_index


def _normalize_timestamp(
    value,
) -> pd.Timestamp:
    """Normalize a single timestamp to timezone-aware UTC."""

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def forecast(
    hours: int | None = None,
    use_stored_history: bool = True,
    observed: pd.DataFrame | None = None,
    future_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Produce the AQI forecast.

    Parameters
    ----------
    use_stored_history
        Splice the feature store's history onto the freshly fetched window.

    observed, future_weather
        Supply both to bypass the network entirely.
    """

    hours = hours or config.FORECAST_HORIZON_HOURS

    model, feature_names = model_registry.load_best_model()

    if observed is None or future_weather is None:
        observed, future_weather = data_sources.fetch_recent_and_forecast(
            past_days=21,
            forecast_days=config.FORECAST_HORIZON_DAYS,
        )

    # ---------------------------------------------------------
    # Normalize live observed data before feature processing.
    # ---------------------------------------------------------
    if observed is not None and not observed.empty:
        observed = observed.copy()

        observed.index = _normalize_datetime_index(
            observed.index,
            name=observed.index.name or "timestamp",
        )

    # ---------------------------------------------------------
    # Normalize future weather timestamps.
    # ---------------------------------------------------------
    if future_weather is not None and not future_weather.empty:
        future_weather = future_weather.copy()

        future_weather.index = _normalize_datetime_index(
            future_weather.index,
            name=future_weather.index.name or "timestamp",
        )

    # ---------------------------------------------------------
    # Add stored history.
    # ---------------------------------------------------------
    if use_stored_history:
        observed = _splice_stored_history(observed)

    # ---------------------------------------------------------
    # Build inference frame.
    # ---------------------------------------------------------
    frame = feature_engineering.build_inference_frame(
        observed,
        future_weather,
        horizons=range(1, hours + 1),
    )

    # ---------------------------------------------------------
    # Make sure inference frame timestamps are UTC.
    # ---------------------------------------------------------
    if "target_time" in frame.columns:
        frame["target_time"] = pd.to_datetime(
            frame["target_time"],
            utc=True,
        )

    if "origin" in frame.columns:
        frame["origin"] = pd.to_datetime(
            frame["origin"],
            utc=True,
        )

    missing = [
        c
        for c in feature_names
        if c not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Feature mismatch: the production model expects {len(missing)} "
            f"column(s) the inference frame does not provide, "
            f"e.g. {missing[:5]}. "
            "Retrain with `python -m src.train` so model and features agree."
        )

    X = frame[feature_names].astype("float32")

    predictions = np.asarray(
        model.predict(X),
        dtype=float,
    )

    result = pd.DataFrame(
        {
            "timestamp": frame["target_time"].values,
            "predicted_aqi": predictions,
            "horizon_h": frame["horizon_h"].values,
        }
    )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    )

    result = (
        result
        .set_index("timestamp")
        .sort_index()
    )

    result["category"] = [
        aqi_category(v)[0]
        for v in result["predicted_aqi"]
    ]

    result["color"] = [
        aqi_category(v)[1]
        for v in result["predicted_aqi"]
    ]

    result["forecast_origin"] = _normalize_timestamp(
        frame["origin"].iloc[0]
    )

    _attach_uncertainty(result)

    config.PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        config.PREDICTIONS_PATH
    )

    logger.info(
        "Forecast: %d hours from %s, range %.0f-%.0f AQI",
        len(result),
        result["forecast_origin"].iloc[0],
        result["predicted_aqi"].min(),
        result["predicted_aqi"].max(),
    )

    return result


def _splice_stored_history(
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepend stored history so the long lag and rolling windows are populated.

    All datetime indexes are normalized to UTC before any comparison.
    """

    if observed is None or observed.empty:
        return observed

    try:
        history = feature_store.read_observations()

    except FileNotFoundError:
        logger.warning(
            "No stored history; relying on the fetched window alone"
        )
        return observed

    except Exception as exc:
        logger.warning(
            "Could not read stored history; relying on fetched window: %s",
            exc,
        )
        return observed

    if history is None or history.empty:
        return observed

    history = history.copy()
    observed = observed.copy()

    # ---------------------------------------------------------
    # Normalize HISTORY index to UTC.
    # ---------------------------------------------------------
    history.index = _normalize_datetime_index(
        history.index,
        name=history.index.name or observed.index.name or "timestamp",
    )

    # ---------------------------------------------------------
    # Normalize OBSERVED index to UTC.
    # ---------------------------------------------------------
    observed.index = _normalize_datetime_index(
        observed.index,
        name=observed.index.name or history.index.name or "timestamp",
    )

    shared = [
        c
        for c in observed.columns
        if c in history.columns
    ]

    if not shared:
        logger.warning(
            "No shared columns between stored history and observed data."
        )
        return observed

    # ---------------------------------------------------------
    # Find the first live observation.
    # Both sides are now UTC-aware, so this comparison is safe.
    # ---------------------------------------------------------
    first_observed = observed.index.min()

    older = history.loc[
        history.index < first_observed,
        shared,
    ]

    # ---------------------------------------------------------
    # Combine historical + current observations.
    # ---------------------------------------------------------
    combined = pd.concat(
        [
            older,
            observed[shared],
        ],
        axis=0,
    )

    # Remove duplicate timestamps.
    combined = combined[
        ~combined.index.duplicated(
            keep="last"
        )
    ]

    # Sort chronologically.
    combined = combined.sort_index()

    # ---------------------------------------------------------
    # Nothing beyond the fetched window's end may survive.
    # ---------------------------------------------------------
    last_observed = observed.index.max()

    combined = combined.loc[
        combined.index <= last_observed
    ]

    return combined


def _attach_uncertainty(
    result: pd.DataFrame,
) -> None:
    """
    Attach a per-horizon uncertainty band from the backtest error curve.
    """

    lower = result["predicted_aqi"].copy()
    upper = result["predicted_aqi"].copy()

    try:
        curve = (
            pd.read_parquet(
                config.HORIZON_METRICS_PATH
            )
            .set_index("horizon_h")["rmse"]
        )

        sigma = result["horizon_h"].map(curve)

        # Fill missing horizons by interpolating.
        sigma = (
            sigma
            .interpolate()
            .bfill()
            .ffill()
        )

    except (
        FileNotFoundError,
        KeyError,
        OSError,
    ):
        logger.info(
            "No horizon metrics yet - uncertainty bands unavailable"
        )

        result["lower_80"] = np.nan
        result["upper_80"] = np.nan

        return

    # 1.28 sigma ≈ 80% interval under normal error assumption.
    margin = (
        1.28
        * sigma.to_numpy(dtype=float)
    )

    result["lower_80"] = np.clip(
        lower.to_numpy() - margin,
        0,
        500,
    )

    result["upper_80"] = np.clip(
        upper.to_numpy() + margin,
        0,
        500,
    )


def daily_summary(
    pred_df: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse hourly forecast into dashboard day cards."""

    daily = pred_df.copy()

    # Make sure timestamp handling works with both
    # timezone-aware and timezone-naive indexes.
    timestamp_index = _normalize_datetime_index(
        daily.index,
        name=daily.index.name or "timestamp",
    )

    daily.index = timestamp_index

    daily["date"] = timestamp_index.date

    summary = (
        daily
        .groupby("date")["predicted_aqi"]
        .agg(["min", "mean", "max"])
        .reset_index()
    )

    summary.columns = [
        "date",
        "min_aqi",
        "avg_aqi",
        "max_aqi",
    ]

    summary["category"] = [
        aqi_category(v)[0]
        for v in summary["avg_aqi"]
    ]

    summary["color"] = [
        aqi_category(v)[1]
        for v in summary["avg_aqi"]
    ]

    summary["advice"] = [
        health_advice(v)
        for v in summary["avg_aqi"]
    ]

    peak_times = (
        daily
        .groupby("date")["predicted_aqi"]
        .idxmax()
    )

    summary["peak_hour"] = [
        pd.Timestamp(ts).strftime("%H:%M")
        for ts in peak_times
    ]

    return summary


def current_conditions(
    max_age_hours: int = 3,
) -> dict:
    """
    Latest observed AQI and pollutant breakdown.

    Handles both timezone-aware and timezone-naive timestamps safely.
    """

    history = feature_store.read_observations()

    if history.empty:
        raise ValueError(
            "No observations available in the feature store."
        )

    # ---------------------------------------------------------
    # Make sure the history index is chronological and UTC.
    # ---------------------------------------------------------
    history = history.copy()

    history.index = _normalize_datetime_index(
        history.index,
        name=history.index.name or "timestamp",
    )

    history = history.sort_index()

    latest = history.iloc[-1]

    # ---------------------------------------------------------
    # Latest observation timestamp
    # ---------------------------------------------------------
    observed_at = _normalize_timestamp(
        history.index.max()
    )

    # ---------------------------------------------------------
    # Current AQI
    # ---------------------------------------------------------
    aqi = float(
        latest[config.TARGET_COLUMN]
    )

    label, colour = aqi_category(aqi)

    # ---------------------------------------------------------
    # Calculate observation age
    # ---------------------------------------------------------
    now_utc = (
        pd.Timestamp.now(tz="UTC")
        .floor("h")
    )

    age_hours = (
        now_utc - observed_at
    ).total_seconds() / 3600.0

    # ---------------------------------------------------------
    # Helper for pollutant/weather values
    # ---------------------------------------------------------
    def _value(column):
        if (
            column in history.columns
            and pd.notna(latest[column])
        ):
            return float(latest[column])

        return None

    # ---------------------------------------------------------
    # Return current conditions
    # ---------------------------------------------------------
    return {
        "timestamp": str(observed_at),
        "age_hours": round(
            max(age_hours, 0.0),
            1,
        ),
        "stale": bool(
            age_hours > max_age_hours
        ),
        "city": config.CITY_NAME,
        "aqi": aqi,
        "category": label,
        "color": colour,
        "advice": health_advice(aqi),
        "pollutants": {
            c: _value(c)
            for c in data_sources.POLLUTANT_COLUMNS
            if c in history.columns
        },
        "temperature_2m": _value(
            "temperature_2m"
        ),
        "wind_speed_10m": _value(
            "wind_speed_10m"
        ),
    }


def load_cached_forecast() -> pd.DataFrame | None:
    """Most recent forecast written to disk, if one exists."""

    if not config.PREDICTIONS_PATH.exists():
        return None

    try:
        result = pd.read_parquet(
            config.PREDICTIONS_PATH
        )

        if result is None or result.empty:
            return None

        # Normalize cached forecast index.
        result.index = _normalize_datetime_index(
            result.index,
            name=result.index.name or "timestamp",
        )

        # Normalize forecast origin if present.
        if "forecast_origin" in result.columns:
            result["forecast_origin"] = pd.to_datetime(
                result["forecast_origin"],
                utc=True,
            )

        return result

    except (
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        logger.warning(
            "Could not load cached forecast: %s",
            exc,
        )
        return None


def main() -> None:
    """Run a forecast from the command line."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
    )

    result = forecast()

    print(
        result[
            [
                "predicted_aqi",
                "category",
                "lower_80",
                "upper_80",
            ]
        ].to_string()
    )

    print()

    print(
        daily_summary(result).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()