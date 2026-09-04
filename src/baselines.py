"""
Reference forecasts.

A regression score means nothing on its own. R^2 = 0.85 on an AQI series sounds
strong until you notice that simply repeating the last observed value scores
0.83 - at which point the model has bought almost nothing. Every headline number
this project reports is therefore accompanied by a *skill score* against these
three baselines, which is the convention operational forecasting uses.

  persistence      ŷ(t+h) = aqi(t)
                   Hard to beat at short lead times; the honest yardstick.

  seasonal naive   ŷ(t+h) = aqi at the same clock hour, the most recent day
                   at or before the origin. Captures the daily cycle for free.

  climatology      ŷ(t+h) = historical mean for that (month, hour) cell,
                   fitted on training data only. The "know nothing about today"
                   forecast, and the reference a useful model must clear easily.

All three are computed from the raw series rather than from the feature matrix,
so they stay exact for any horizon rather than being limited to whichever lags
happen to be materialised as columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence(raw_target: pd.Series, origins: pd.Series) -> np.ndarray:
    """Last observed value at the forecast origin."""
    return raw_target.reindex(pd.DatetimeIndex(origins)).to_numpy(dtype=float)


def seasonal_naive(raw_target: pd.Series,
                   origins: pd.Series,
                   horizons: pd.Series,
                   period_hours: int = 24) -> np.ndarray:
    """
    Value at the same clock hour on the most recent complete cycle.

    For h <= 24 that is "same hour yesterday"; for h = 50 it is "same hour two
    days back", and so on - always the newest observation at or before the
    origin that shares the target's position in the daily cycle.
    """
    origins = pd.DatetimeIndex(origins)
    horizons = np.asarray(horizons, dtype=float)

    cycles_back = np.ceil(horizons / period_hours)
    offset_hours = cycles_back * period_hours - horizons  # in [0, period_hours)
    lookup = origins - pd.to_timedelta(offset_hours, unit="h")

    return raw_target.reindex(lookup).to_numpy(dtype=float)


class Climatology:
    """
    Mean AQI per (month, hour-of-day) cell.

    Fitted on training rows only - fitting on the full series would leak the
    test period's seasonal level into the baseline and understate the model's
    advantage over it.
    """

    def __init__(self, min_samples: int = 3):
        self.min_samples = min_samples
        self.table_: pd.Series | None = None
        self.global_mean_: float = np.nan

    def fit(self, raw_target: pd.Series) -> "Climatology":
        series = raw_target.dropna()
        if series.empty:
            raise ValueError("Cannot fit climatology on an empty series")

        frame = pd.DataFrame({
            "value": series.values,
            "month": series.index.month,
            "hour": series.index.hour,
        })
        grouped = frame.groupby(["month", "hour"])["value"]
        table = grouped.mean()
        counts = grouped.size()

        # Thin cells fall back to the global mean rather than trusting one sample.
        self.table_ = table[counts >= self.min_samples]
        self.global_mean_ = float(series.mean())
        return self

    def predict(self, target_times) -> np.ndarray:
        if self.table_ is None:
            raise RuntimeError("Climatology.fit must be called before predict")
        idx = pd.DatetimeIndex(target_times)
        keys = pd.MultiIndex.from_arrays([idx.month, idx.hour])
        values = self.table_.reindex(keys).to_numpy(dtype=float)
        return np.where(np.isnan(values), self.global_mean_, values)


def compute_all(raw_target: pd.Series,
                dataset: pd.DataFrame,
                climatology: Climatology | None = None) -> dict:
    """
    Every baseline's predictions for the rows of a supervised `dataset`.

    Returns a name -> prediction-array mapping aligned to `dataset`'s row order.
    """
    preds = {
        "persistence": persistence(raw_target, dataset["origin"]),
        "seasonal_naive_24h": seasonal_naive(raw_target, dataset["origin"],
                                             dataset["horizon_h"]),
    }
    if climatology is not None:
        preds["climatology"] = climatology.predict(dataset["target_time"])

    # A missing lookup means the history had a gap there; persistence is the
    # safest stand-in, and it keeps array lengths aligned for scoring.
    fallback = preds["persistence"]
    for name, values in preds.items():
        preds[name] = np.where(np.isnan(values), fallback, values)

    return preds


def fit_climatology(raw_target: pd.Series, train_end) -> Climatology:
    """Fit climatology using only observations up to `train_end`."""
    train_slice = raw_target[raw_target.index <= pd.Timestamp(train_end)]
    return Climatology().fit(train_slice)


BASELINE_NAMES = ("persistence", "seasonal_naive_24h", "climatology")


def reference_name() -> str:
    """
    The baseline that headline skill scores are quoted against.

    Persistence is the strictest of the three at the lead times we care about,
    so beating it is the claim worth making.
    """
    return "persistence"


__all__ = ["persistence", "seasonal_naive", "Climatology", "compute_all",
           "fit_climatology", "BASELINE_NAMES", "reference_name"]
