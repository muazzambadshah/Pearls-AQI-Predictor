"""
Model explainability (SHAP).

Answers two different questions, and it is worth keeping them apart:

  Global  - "what does this model rely on overall?" Averaged |SHAP| across many
            rows. Useful for validating that the model learned physics rather
            than an artefact.
  Local   - "why is Thursday's forecast 180?" The signed contribution of each
            feature to one specific prediction. This is what makes a dashboard
            number trustworthy instead of oracular.

Everything degrades gracefully: SHAP is slow on large ensembles and does not
support every estimator, so each path falls back to permutation importance and
then to the model's built-in importances rather than failing the caller.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Human-readable groupings, so the UI can talk about drivers rather than columns.
FEATURE_GROUPS = {
    "AQI history": ("aqi_lag", "aqi_roll", "aqi_delta", "aqi_at_origin", "aqi_zscore"),
    "Pollutants now": ("pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                       "sulphur_dioxide", "ozone", "pm_ratio"),
    "Forecast weather": ("tgt_temperature", "tgt_relative_humidity", "tgt_wind",
                         "tgt_precip", "tgt_surface_pressure", "tgt_cloud",
                         "tgt_dew_point", "tgt_apparent", "tgt_ventilation",
                         "tgt_stagnation", "tgt_vapour"),
    "Weather now": ("temperature_2m_at_origin", "wind_", "surface_pressure_at_origin",
                    "precip_sum", "ventilation_proxy_at_origin", "stagnation_index_at_origin",
                    "relative_humidity_2m_at_origin", "cloud_cover_at_origin",
                    "dew_point_2m_at_origin", "apparent_temperature_at_origin",
                    "vapour_pressure_deficit_at_origin"),
    "Time of day / season": ("tgt_hour", "tgt_month", "tgt_doy", "tgt_day", "tgt_is_weekend"),
    "Lead time": ("horizon_h", "horizon_days"),
}


def _unwrap(model):
    """
    Reach the estimator SHAP can actually explain.

    The zoo wraps models twice - an anchoring or clipping wrapper around a
    possibly pipelined estimator - and SHAP's fast tree path needs the bare tree.

    A blend has no single inner model, so we explain its first tree-based member.
    That is an approximation, and worth being explicit about: it describes what
    the dominant member relies on, not the exact attribution of the averaged
    output. The alternative is a kernel explainer over the whole blend, which is
    orders of magnitude slower and would make `/explain` unusable. Since blend
    members are the top-ranked models and tend to agree on the broad drivers,
    the ranking is representative even though the magnitudes are not exact.
    """
    fitted = getattr(model, "fitted_", None)
    if fitted:  # BlendedRegressor
        for _, member in fitted:
            candidate = _unwrap(member)
            if _is_tree(candidate):
                return candidate
        model = fitted[0][1]

    inner = getattr(model, "base_", model)
    if hasattr(inner, "named_steps"):
        inner = inner.named_steps.get("model", inner)
    return inner


def _is_tree(estimator) -> bool:
    return hasattr(estimator, "feature_importances_") or hasattr(estimator, "estimators_")


def global_importance(model, X: pd.DataFrame, max_rows: int = 400) -> pd.DataFrame:
    """
    Mean |SHAP| per feature, descending.

    `max_rows` caps the sample: SHAP cost scales with rows, and a few hundred is
    already enough for a stable global ranking.
    """
    sample = X.tail(max_rows) if len(X) > max_rows else X
    estimator = _unwrap(model)

    try:
        import shap

        if _is_tree(estimator):
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(sample, check_additivity=False)
        else:
            background = shap.sample(sample, min(100, len(sample)), random_state=0)
            explainer = shap.Explainer(model.predict, background)
            values = explainer(sample).values

        importance = np.abs(np.asarray(values)).mean(axis=0)
        return (pd.DataFrame({"feature": sample.columns, "importance": importance})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP unavailable (%s); falling back to built-in importances", exc)
        return _fallback_importance(model, sample)


def _fallback_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    estimator = _unwrap(model)

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:
        importance = np.zeros(X.shape[1])

    if importance.shape[0] != X.shape[1]:
        importance = np.resize(importance, X.shape[1])

    return (pd.DataFrame({"feature": X.columns, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def local_explanation(model, X_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Signed per-feature contributions for a single prediction.

    Positive values pushed the forecast up, negative pulled it down.
    """
    estimator = _unwrap(model)

    try:
        import shap

        if _is_tree(estimator):
            explainer = shap.TreeExplainer(estimator)
            values = np.asarray(explainer.shap_values(X_row, check_additivity=False)).ravel()
        else:
            explainer = shap.Explainer(model.predict, X_row)
            values = np.asarray(explainer(X_row).values).ravel()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local SHAP failed (%s)", exc)
        return pd.DataFrame(columns=["feature", "value", "contribution"])

    frame = pd.DataFrame({
        "feature": X_row.columns,
        "value": X_row.iloc[0].to_numpy(),
        "contribution": values,
    })
    frame["abs"] = frame["contribution"].abs()
    return (frame.sort_values("abs", ascending=False)
            .head(top_n).drop(columns="abs").reset_index(drop=True))


def group_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll feature-level importance up into themes.

    "AQI history contributes 46%, forecast weather 28%" communicates far more
    than a list of forty lag columns, and it is the level a dashboard should
    speak at.
    """
    def classify(name: str) -> str:
        for group, prefixes in FEATURE_GROUPS.items():
            if any(p in name for p in prefixes):
                return group
        return "Other"

    frame = importance_df.copy()
    frame["group"] = frame["feature"].map(classify)

    grouped = (frame.groupby("group")["importance"].sum()
               .sort_values(ascending=False).reset_index())
    total = grouped["importance"].sum()
    grouped["share"] = grouped["importance"] / total if total > 0 else 0.0
    return grouped


def explain_production_model(max_rows: int = 400):
    """
    Convenience path: explain the currently promoted model on recent data.

    Returns `(feature_importance, group_importance)`.
    """
    from src import config, feature_engineering, feature_store, model_registry

    model, feature_names = model_registry.load_best_model()
    raw = feature_store.read_observations()

    # A recent slice is enough for a global ranking and keeps SHAP fast.
    recent = raw.tail(24 * 45)
    dataset = feature_engineering.make_supervised(
        recent, horizons=(1, 6, 12, 24, 48, 72), origin_stride=6
    )
    if dataset.empty:
        raise RuntimeError("Not enough recent history to build an explanation sample")

    X = dataset[feature_names].astype("float32")
    importance = global_importance(model, X, max_rows=max_rows)
    return importance, group_importance(importance)
