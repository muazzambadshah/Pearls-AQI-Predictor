"""
Pipeline tests - feature engineering, models, evaluation, alerts, inference.

Everything here runs offline against the deterministic synthetic generator, so
CI never depends on a live API or on the state of the feature store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import (alerts, baselines, config, data_sources, evaluate,
                 feature_engineering as fe, models, predict)


@pytest.fixture(scope="module")
def raw():
    return data_sources.generate_synthetic_data(days=90, seed=11)


@pytest.fixture(scope="module")
def dataset(raw):
    return fe.make_supervised(raw, horizons=(1, 6, 24, 48, 72), origin_stride=4)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def test_synthetic_data_is_deterministic():
    a = data_sources.generate_synthetic_data(days=10, seed=3)
    b = data_sources.generate_synthetic_data(days=10, seed=3)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_synthetic_data_has_expected_shape_and_columns():
    df = data_sources.generate_synthetic_data(days=10)
    assert len(df) == 240
    assert config.TARGET_COLUMN in df.columns
    for col in fe.FUTURE_WEATHER:
        assert col in df.columns, f"synthetic data is missing {col}"
    assert df.isna().sum().sum() == 0


def test_us_aqi_from_pm25_matches_epa_breakpoints():
    # Breakpoint anchors from the 2024 EPA revision.
    assert data_sources.us_aqi_from_pm25([0.0])[0] == pytest.approx(0.0, abs=0.5)
    assert data_sources.us_aqi_from_pm25([9.0])[0] == pytest.approx(50.0, abs=0.5)
    assert data_sources.us_aqi_from_pm25([35.4])[0] == pytest.approx(100.0, abs=0.5)
    assert data_sources.us_aqi_from_pm25([55.4])[0] == pytest.approx(150.0, abs=0.5)
    assert data_sources.us_aqi_from_pm25([600.0])[0] == 500.0


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def test_supervised_dataset_is_complete_and_typed(dataset):
    assert not dataset.empty
    X = dataset[fe.feature_columns(dataset)]
    assert X.isna().sum().sum() == 0, "feature matrix must not contain NaN"
    assert dataset["y"].notna().all()
    assert set(dataset["horizon_h"].unique()) == {1, 6, 24, 48, 72}


def test_feature_columns_exclude_bookkeeping(dataset):
    cols = fe.feature_columns(dataset)
    for meta in fe.META_COLUMNS:
        assert meta not in cols


def test_cyclical_encodings_are_bounded(raw):
    feats = fe.build_future_features(raw[list(fe.FUTURE_WEATHER)])
    for col in ("tgt_hour_sin", "tgt_hour_cos", "tgt_month_sin", "tgt_doy_cos"):
        assert feats[col].between(-1.0, 1.0).all()


def test_hour_23_and_hour_0_are_adjacent_in_cyclical_space(raw):
    """The whole reason for sin/cos encoding: midnight must neighbour 23:00."""
    feats = fe.build_future_features(raw[list(fe.FUTURE_WEATHER)])
    h23 = feats[feats["tgt_hour"] == 23].iloc[0]
    h00 = feats[feats["tgt_hour"] == 0].iloc[0]
    h12 = feats[feats["tgt_hour"] == 12].iloc[0]

    def dist(a, b):
        return np.hypot(a["tgt_hour_sin"] - b["tgt_hour_sin"],
                        a["tgt_hour_cos"] - b["tgt_hour_cos"])

    assert dist(h23, h00) < dist(h23, h12)


def test_wind_vector_decomposition_preserves_speed(raw):
    feats = fe.build_origin_features(raw)
    speed = np.hypot(feats["wind_u_10m_at_origin"], feats["wind_v_10m_at_origin"])
    np.testing.assert_allclose(speed.to_numpy(), raw["wind_speed_10m"].to_numpy(), rtol=1e-5)


def test_origin_stride_reduces_origins_without_dropping_horizons(raw):
    dense = fe.make_supervised(raw, horizons=(1, 24), origin_stride=1)
    sparse = fe.make_supervised(raw, horizons=(1, 24), origin_stride=6)
    assert sparse["origin"].nunique() < dense["origin"].nunique()
    assert set(sparse["horizon_h"].unique()) == set(dense["horizon_h"].unique())


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def test_persistence_returns_the_origin_value(raw, dataset):
    target = raw[config.TARGET_COLUMN]
    preds = baselines.persistence(target, dataset["origin"])
    np.testing.assert_allclose(preds, dataset[fe.ANCHOR_COLUMN].to_numpy(dtype=float), rtol=1e-5)


def test_seasonal_naive_looks_back_a_whole_number_of_days(raw, dataset):
    target = raw[config.TARGET_COLUMN]
    preds = baselines.seasonal_naive(target, dataset["origin"], dataset["horizon_h"])
    assert np.isfinite(preds).mean() > 0.9

    # For h=24 the lookup is exactly the origin itself.
    h24 = dataset[dataset["horizon_h"] == 24]
    expected = target.reindex(pd.DatetimeIndex(h24["origin"])).to_numpy()
    got = baselines.seasonal_naive(target, h24["origin"], h24["horizon_h"])
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_climatology_is_fitted_only_on_training_data(raw):
    target = raw[config.TARGET_COLUMN]
    cutoff = raw.index[len(raw) // 2]
    clim = baselines.fit_climatology(target, cutoff)

    corrupted = target.copy()
    corrupted[corrupted.index > cutoff] = 480.0
    clim_corrupt = baselines.fit_climatology(corrupted, cutoff)

    sample = raw.index[-50:]
    np.testing.assert_allclose(clim.predict(sample), clim_corrupt.predict(sample), rtol=1e-9)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def test_regression_metrics_on_a_perfect_forecast():
    y = np.array([10.0, 50.0, 120.0, 300.0])
    m = evaluate.regression_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(0.0)


def test_regression_metrics_ignore_nan_pairs():
    y_true = np.array([10.0, np.nan, 30.0])
    y_pred = np.array([12.0, 20.0, np.nan])
    m = evaluate.regression_metrics(y_true, y_pred)
    assert m["n"] == 1


def test_skill_score_semantics():
    assert evaluate.skill_score(5.0, 10.0) == pytest.approx(0.5)
    assert evaluate.skill_score(10.0, 10.0) == pytest.approx(0.0)
    assert evaluate.skill_score(20.0, 10.0) == pytest.approx(-1.0)


def test_aqi_band_index_boundaries():
    bands = evaluate.aqi_band_index([25, 75, 125, 175, 250, 400])
    assert list(bands) == [0, 1, 2, 3, 4, 5]


def test_operational_metrics_detect_exceedances():
    y_true = np.array([100.0, 200.0, 160.0, 50.0])
    y_pred = np.array([110.0, 190.0, 140.0, 60.0])
    m = evaluate.operational_metrics(y_true, y_pred, threshold=150.0)
    # Two true exceedances (200, 160); model flags only the 190.
    assert m["exceedance_recall"] == pytest.approx(0.5)
    assert m["exceedance_precision"] == pytest.approx(1.0)


def test_per_horizon_metrics_produce_one_row_per_horizon(dataset):
    preds = dataset[fe.ANCHOR_COLUMN].to_numpy(dtype=float)
    frame = evaluate.per_horizon_metrics(dataset, preds)
    assert len(frame) == dataset["horizon_h"].nunique()
    assert frame["horizon_h"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def test_anchored_regressor_reconstructs_the_level(dataset):
    from sklearn.linear_model import Ridge

    X, y, anchor = fe.split_feature_matrix(dataset)
    model = models.AnchoredRegressor(Ridge(alpha=1.0)).fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.isfinite(preds).all()
    # An anchored model fitted and scored on the same rows should beat the
    # anchor it was built on.
    assert evaluate.regression_metrics(y, preds)["rmse"] < \
           evaluate.regression_metrics(y, anchor)["rmse"]


def test_predictions_are_clipped_to_the_valid_aqi_range(dataset):
    from sklearn.dummy import DummyRegressor

    X, y, _ = fe.split_feature_matrix(dataset)
    model = models.ClippedRegressor(DummyRegressor(strategy="constant", constant=-500.0))
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds >= models.AQI_MIN).all() and (preds <= models.AQI_MAX).all()


def test_blend_averages_its_members(dataset):
    from sklearn.dummy import DummyRegressor

    X, y, _ = fe.split_feature_matrix(dataset)
    blend = models.BlendedRegressor(members=[
        ("low", models.ClippedRegressor(DummyRegressor(strategy="constant", constant=100.0))),
        ("high", models.ClippedRegressor(DummyRegressor(strategy="constant", constant=200.0))),
    ]).fit(X, y)

    np.testing.assert_allclose(blend.predict(X), 150.0)


def test_blend_respects_explicit_weights(dataset):
    from sklearn.dummy import DummyRegressor

    X, y, _ = fe.split_feature_matrix(dataset)
    blend = models.BlendedRegressor(
        members=[
            ("low", models.ClippedRegressor(DummyRegressor(strategy="constant", constant=100.0))),
            ("high", models.ClippedRegressor(DummyRegressor(strategy="constant", constant=200.0))),
        ],
        weights=[3.0, 1.0],   # normalised to 0.75 / 0.25
    ).fit(X, y)

    np.testing.assert_allclose(blend.predict(X), 125.0)


def test_blend_is_cloneable_so_the_backtest_can_refit_it(dataset):
    from sklearn.base import clone
    from sklearn.linear_model import Ridge

    X, y, _ = fe.split_feature_matrix(dataset)
    blend = models.BlendedRegressor(members=[
        ("a", models.ClippedRegressor(Ridge(alpha=1.0))),
        ("b", models.ClippedRegressor(Ridge(alpha=10.0))),
    ])
    fitted = clone(blend).fit(X.head(2000), y.head(2000))
    preds = fitted.predict(X.head(2000))
    assert preds.shape == (2000,)
    assert np.isfinite(preds).all()


def test_model_requirements_detects_nested_dependencies():
    """
    A pickle silently carries the import graph of whatever produced it, so the
    registry records what a model needs in order to load. Missing this is how a
    shipped artefact ends up unloadable in a clean environment.
    """
    from sklearn.linear_model import Ridge

    plain = models.ClippedRegressor(Ridge())
    assert models.model_requirements_probe(plain) == []

    if models.HAS_LIGHTGBM:
        blend = models.BlendedRegressor(members=[
            ("a", models.ClippedRegressor(Ridge())),
            ("b", models.build_model_zoo(fast=True)["lightgbm"]),
        ])
        assert "lightgbm" in models.model_requirements_probe(blend)


def test_blend_rejects_an_empty_member_list(dataset):
    X, y, _ = fe.split_feature_matrix(dataset)
    with pytest.raises(ValueError, match="at least one member"):
        models.BlendedRegressor(members=[]).fit(X, y)


def test_model_zoo_covers_the_required_families():
    zoo = models.build_model_zoo(fast=True)
    assert "ridge" in zoo, "a linear/statistical candidate is required"
    assert "random_forest" in zoo, "a tree-ensemble candidate is required"
    assert any("gradient_boosting" in k or "lightgbm" in k for k in zoo), \
        "a boosted candidate is required"
    assert len(zoo) >= 6


def test_every_zoo_model_fits_and_predicts(dataset):
    X, y, _ = fe.split_feature_matrix(dataset)
    X, y = X.head(3000), y.head(3000)

    for name, model in models.build_model_zoo(fast=True).items():
        from sklearn.base import clone

        fitted = clone(model).fit(X, y)
        preds = fitted.predict(X)
        assert preds.shape == y.shape, f"{name} returned the wrong shape"
        assert np.isfinite(preds).all(), f"{name} produced non-finite predictions"


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
def test_backtest_folds_are_chronological_and_expanding(dataset):
    from src import backtest

    cutoffs = [c for c, _ in backtest.make_folds(dataset, n_folds=3)]
    assert cutoffs == sorted(cutoffs)

    sizes = []
    for cutoff, end in backtest.make_folds(dataset, n_folds=3):
        train_df, _ = backtest.split_at(dataset, cutoff, end)
        sizes.append(len(train_df))
    assert sizes == sorted(sizes), "training window should grow with each fold"


def test_backtest_runs_end_to_end(dataset, raw):
    from sklearn.linear_model import Ridge
    from src import backtest

    model = models.ClippedRegressor(Ridge(alpha=1.0))
    folds, preds = backtest.backtest_model(
        model, dataset, raw[config.TARGET_COLUMN], n_folds=2,
        model_name="ridge_test", collect_predictions=True,
    )
    assert len(folds) == 2
    assert {"rmse", "mae", "r2", "skill_vs_persistence"} <= set(folds.columns)
    assert preds is not None and len(preds) > 0


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def _forecast_frame(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="h", name="timestamp")
    return pd.DataFrame({"predicted_aqi": values}, index=idx)


def test_alerts_group_consecutive_hours_into_one_episode():
    frame = _forecast_frame([50] * 5 + [180] * 6 + [60] * 5)
    episodes = alerts.find_episodes(frame, threshold=150)
    assert len(episodes) == 1
    assert episodes[0]["duration_hours"] == 6
    assert episodes[0]["peak_aqi"] == pytest.approx(180.0)


def test_alerts_ignore_single_hour_blips():
    frame = _forecast_frame([50] * 5 + [180] + [50] * 5)
    assert alerts.find_episodes(frame, threshold=150, min_duration_hours=2) == []


def test_alerts_separate_distinct_episodes():
    frame = _forecast_frame([50] * 3 + [200] * 4 + [60] * 6 + [190] * 3)
    episodes = alerts.find_episodes(frame, threshold=150)
    assert len(episodes) == 2


def test_alert_payload_reports_quiet_conditions():
    payload = alerts.build_alert(_forecast_frame([40] * 24), threshold=150)
    assert payload["alert"] is False
    assert payload["episodes"] == []


def test_severity_tiers_escalate():
    assert alerts.severity_for(120) == "advisory"
    assert alerts.severity_for(160) == "warning"
    assert alerts.severity_for(220) == "critical"
    assert alerts.severity_for(350) == "emergency"


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def test_aqi_category_boundaries():
    assert predict.aqi_category(25)[0] == "Good"
    assert predict.aqi_category(75)[0] == "Moderate"
    assert predict.aqi_category(125)[0] == "Unhealthy for Sensitive Groups"
    assert predict.aqi_category(175)[0] == "Unhealthy"
    assert predict.aqi_category(250)[0] == "Very Unhealthy"
    assert predict.aqi_category(450)[0] == "Hazardous"


def test_health_advice_is_defined_for_every_band():
    for aqi in (10, 75, 125, 175, 250, 450):
        assert isinstance(predict.health_advice(aqi), str)
        assert len(predict.health_advice(aqi)) > 20


def test_daily_summary_aggregates_per_calendar_day():
    idx = pd.date_range("2026-01-01", periods=72, freq="h", name="timestamp")
    frame = pd.DataFrame({"predicted_aqi": np.linspace(60, 200, 72)}, index=idx)
    summary = predict.daily_summary(frame)
    assert len(summary) == 3
    assert {"date", "min_aqi", "avg_aqi", "max_aqi", "category", "advice",
            "peak_hour"} <= set(summary.columns)
    assert (summary["min_aqi"] <= summary["avg_aqi"]).all()
    assert (summary["avg_aqi"] <= summary["max_aqi"]).all()


def test_inference_frame_matches_training_schema(raw):
    """Train and serve must produce the identical feature set, or serving breaks."""
    observed = raw.iloc[:-72]
    future_weather = raw.iloc[-72:][list(fe.FUTURE_WEATHER)]

    train_ds = fe.make_supervised(raw, horizons=(1, 24, 72), origin_stride=8)
    infer = fe.build_inference_frame(observed, future_weather, horizons=(1, 24, 72))

    train_cols = set(fe.feature_columns(train_ds))
    infer_cols = set(fe.feature_columns(infer))
    assert train_cols == infer_cols, (
        f"schema drift - only in training: {sorted(train_cols - infer_cols)[:5]}, "
        f"only in inference: {sorted(infer_cols - train_cols)[:5]}"
    )


def test_inference_frame_rejects_insufficient_history(raw):
    with pytest.raises(ValueError, match="history"):
        fe.build_inference_frame(raw.head(5), raw.tail(72)[list(fe.FUTURE_WEATHER)])
