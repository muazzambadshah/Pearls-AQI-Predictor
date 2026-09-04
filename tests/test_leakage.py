"""
Leakage guards.

These are the most important tests in the project. The original implementation
of this forecaster reported R^2 = 1.00 because `aqi_change_rate = aqi.diff()`
and `aqi_lag_1h` were both handed to the model as inputs - and

    aqi_lag_1h + aqi_change_rate == aqi[t]

so the target was an exact sum of two features. The model had nothing to learn
and every metric was meaningless. Nothing in the code looked obviously wrong;
only the suspiciously perfect score gave it away.

The tests below make that class of bug impossible to reintroduce silently. The
central one is `test_features_do_not_depend_on_the_target`: it perturbs future
AQI values and asserts the feature matrix comes back bit-identical. If any
feature reads the target at or after the prediction time, that test fails - no
matter how the leak is spelled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data_sources, feature_engineering as fe


@pytest.fixture(scope="module")
def raw():
    return data_sources.generate_synthetic_data(days=60, seed=7)


@pytest.fixture(scope="module")
def dataset(raw):
    return fe.make_supervised(raw, horizons=(1, 6, 24, 72), origin_stride=3)


def test_features_do_not_depend_on_the_target(raw):
    """
    Corrupt the AQI series *after* a cutoff and require the feature matrix for
    origins before that cutoff to be unchanged.

    This is a direct, spelling-independent proof of the no-leakage property: if
    any feature reads AQI at or beyond the target time, corrupting those values
    must change the feature matrix.
    """
    cutoff = raw.index[len(raw) // 2]

    corrupted = raw.copy()
    mask = corrupted.index > cutoff
    corrupted.loc[mask, config.TARGET_COLUMN] = 999.0

    horizons = (1, 6, 24, 72)
    clean_ds = fe.make_supervised(raw, horizons=horizons, origin_stride=5)
    dirty_ds = fe.make_supervised(corrupted, horizons=horizons, origin_stride=5)

    # Compare only origins whose entire feature window predates the cutoff.
    keep_clean = clean_ds[clean_ds["origin"] <= cutoff]
    keep_dirty = dirty_ds[dirty_ds["origin"] <= cutoff]

    common = keep_clean.merge(
        keep_dirty, on=["origin", "horizon_h"], suffixes=("_clean", "_dirty")
    )
    assert len(common) > 50, "Not enough overlapping rows to make the test meaningful"

    # `horizon_h` is a merge key, so it carries no suffix and is identical by
    # construction; every other feature gets the _clean/_dirty pair.
    merge_keys = {"horizon_h"}
    feature_cols = [c for c in fe.feature_columns(clean_ds) if c not in merge_keys]

    mismatched = []
    for col in feature_cols:
        a = common[f"{col}_clean"].to_numpy(dtype=float)
        b = common[f"{col}_dirty"].to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True):
            mismatched.append(col)

    assert not mismatched, (
        f"{len(mismatched)} feature(s) changed when only FUTURE target values were "
        f"corrupted, which means they read the target: {mismatched[:10]}"
    )


def test_target_is_not_reconstructable_from_features(dataset):
    """
    No single feature may be a near-perfect linear stand-in for the target.

    `aqi_at_origin` is allowed to correlate strongly at short lead times - that
    is persistence, and it is legitimate - so this checks the pathological case:
    a feature that matches the target almost exactly at *every* horizon,
    including 72 hours out, which no honest predictor can manage.
    """
    long_range = dataset[dataset["horizon_h"] == 72]
    assert len(long_range) > 30

    y = long_range["y"].to_numpy(dtype=float)
    suspicious = []

    for col in fe.feature_columns(long_range):
        values = long_range[col].to_numpy(dtype=float)
        if np.std(values) == 0 or np.std(y) == 0:
            continue
        corr = abs(np.corrcoef(values, y)[0, 1])
        if corr > 0.99:
            suspicious.append((col, round(float(corr), 4)))

    assert not suspicious, f"Feature(s) essentially equal to the 72h target: {suspicious}"


def test_no_target_time_pollutant_columns(dataset):
    """
    Pollutant readings at the target hour must never appear as features.

    Open-Meteo derives `us_aqi` from the pollutant concentrations, so a target-time
    PM2.5 column would effectively be the answer in different units.
    """
    forbidden = []
    for col in fe.feature_columns(dataset):
        if not col.startswith("tgt_"):
            continue
        if any(p in col for p in data_sources.POLLUTANT_COLUMNS) or "aqi" in col.lower():
            forbidden.append(col)

    assert not forbidden, f"Target-time pollutant/AQI features present: {forbidden}"


def test_origin_features_use_only_past_observations(raw):
    """Truncating history after `t` must not change the features computed at `t`."""
    cutoff_idx = len(raw) - 200
    cutoff = raw.index[cutoff_idx]

    full = fe.build_origin_features(raw)
    truncated = fe.build_origin_features(raw.loc[:cutoff])

    row_full = full.loc[cutoff]
    row_trunc = truncated.loc[cutoff]

    pd.testing.assert_series_equal(
        row_full, row_trunc, check_names=False, rtol=1e-9,
        obj=f"origin features at {cutoff} changed when future rows were removed",
    )


def test_target_time_is_always_origin_plus_horizon(dataset):
    delta = (pd.to_datetime(dataset["target_time"]) - pd.to_datetime(dataset["origin"]))
    expected = pd.to_timedelta(dataset["horizon_h"].astype(float), unit="h")
    assert (delta == expected).all(), "target_time does not equal origin + horizon"


def test_no_duplicate_origin_horizon_pairs(dataset):
    assert not dataset.duplicated(subset=["origin", "horizon_h"]).any()


def test_backtest_split_embargoes_the_test_window(dataset):
    """
    Training rows must not have labels inside the test window.

    Splitting on `origin` alone would let a 72-hour-horizon sample originating
    just before the cutoff carry a label from well after it - training on the
    very future being scored.
    """
    from src import backtest

    cutoff = dataset["origin"].quantile(0.6)
    train_df, test_df = backtest.split_at(dataset, cutoff)

    assert len(train_df) > 0 and len(test_df) > 0
    assert train_df["target_time"].max() <= cutoff, (
        "A training label falls after the cutoff - the embargo is not holding"
    )
    assert test_df["origin"].min() > cutoff
