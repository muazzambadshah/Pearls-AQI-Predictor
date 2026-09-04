"""
Scoring.

Three families of metric, because "how accurate is it?" has three different
answers depending on who is asking:

  Regression      RMSE / MAE / R^2 / bias - what the brief asks for.
  Skill           Improvement over a reference forecast. The number that says
                  whether the model is actually earning its keep.
  Operational     Does it get the AQI *category* right, and does it catch the
                  unhealthy episodes? This is what a dashboard user and an alert
                  subscriber actually experience.

Everything is also reported per horizon, because a single averaged figure hides
the shape that matters most: a 1-hour forecast and a 72-hour forecast are wildly
different problems, and averaging them together flatters the hard one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# US EPA AQI categories.
AQI_BANDS = [
    (0, 50, "Good"),
    (50, 100, "Moderate"),
    (100, 150, "Unhealthy for Sensitive Groups"),
    (150, 200, "Unhealthy"),
    (200, 300, "Very Unhealthy"),
    (300, 501, "Hazardous"),
]

UNHEALTHY_THRESHOLD = 150.0


def aqi_band_index(values) -> np.ndarray:
    """Map AQI values onto category indices 0-5."""
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=int)
    for i, (low, high, _) in enumerate(AQI_BANDS):
        out[(arr >= low) & (arr < high)] = i
    out[arr >= AQI_BANDS[-1][1]] = len(AQI_BANDS) - 1
    return out


def aqi_band_label(value: float) -> str:
    return AQI_BANDS[int(aqi_band_index([value])[0])][2]


def regression_metrics(y_true, y_pred) -> dict:
    """Core regression scores, NaN-safe."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if y_true.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                "bias": float("nan"), "smape": float("nan"), "n": 0}

    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / denom)) * 100)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # R^2 is undefined when the truth has no variance (a single test row).
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 and np.ptp(y_true) > 0
              else float("nan"),
        "bias": float(np.mean(y_pred - y_true)),
        "smape": smape,
        "n": int(y_true.size),
    }


def operational_metrics(y_true, y_pred, threshold: float = UNHEALTHY_THRESHOLD) -> dict:
    """
    How the forecast performs as a decision aid rather than as a number.

    `category_accuracy` is the share of hours placed in the correct EPA band.
    `within_one_band` allows a single-band miss, which is the tolerance most
    public dashboards implicitly work to. The exceedance figures describe alert
    quality at `threshold`.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if y_true.size == 0:
        return {}

    true_band = aqi_band_index(y_true)
    pred_band = aqi_band_index(y_pred)

    actual = y_true >= threshold
    predicted = y_pred >= threshold
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else float("nan"))

    return {
        "category_accuracy": float(np.mean(true_band == pred_band)),
        "within_one_band": float(np.mean(np.abs(true_band - pred_band) <= 1)),
        "exceedance_precision": float(precision),
        "exceedance_recall": float(recall),
        "exceedance_f1": float(f1),
        "exceedance_base_rate": float(np.mean(actual)),
    }


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """
    Fractional RMSE reduction against a reference forecast.

    1.0 is perfect, 0.0 means no better than the reference, negative means worse.
    """
    if not np.isfinite(baseline_rmse) or baseline_rmse == 0:
        return float("nan")
    return float(1.0 - model_rmse / baseline_rmse)


def evaluate_predictions(y_true, y_pred, baseline_preds: dict | None = None,
                         threshold: float = UNHEALTHY_THRESHOLD) -> dict:
    """Full metric bundle for one set of predictions."""
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(operational_metrics(y_true, y_pred, threshold=threshold))

    for name, preds in (baseline_preds or {}).items():
        base = regression_metrics(y_true, preds)
        metrics[f"skill_vs_{name}"] = skill_score(metrics["rmse"], base["rmse"])
    return metrics


def per_horizon_metrics(dataset: pd.DataFrame,
                        y_pred,
                        baseline_preds: dict | None = None,
                        horizon_col: str = "horizon_h",
                        truth_col: str = "y") -> pd.DataFrame:
    """
    Metrics grouped by lead time.

    The resulting curve is the single most informative artefact this project
    produces: it shows exactly where the model stops adding value over
    persistence, which is the question anyone evaluating a forecast asks first.
    """
    frame = pd.DataFrame({
        "horizon_h": dataset[horizon_col].to_numpy(),
        "y_true": dataset[truth_col].to_numpy(dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    for name, preds in (baseline_preds or {}).items():
        frame[f"base_{name}"] = np.asarray(preds, dtype=float)

    rows = []
    for horizon, group in frame.groupby("horizon_h"):
        row = {"horizon_h": float(horizon)}
        row.update(regression_metrics(group["y_true"], group["y_pred"]))
        row.update(operational_metrics(group["y_true"], group["y_pred"]))
        for name in (baseline_preds or {}):
            base = regression_metrics(group["y_true"], group[f"base_{name}"])
            row[f"{name}_rmse"] = base["rmse"]
            row[f"skill_vs_{name}"] = skill_score(row["rmse"], base["rmse"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("horizon_h").reset_index(drop=True)


def daily_metrics(dataset: pd.DataFrame, y_pred, truth_col: str = "y") -> pd.DataFrame:
    """
    Accuracy of the day-1 / day-2 / day-3 aggregates the dashboard shows.

    Users read "tomorrow's average AQI", not hour 37, so this is closer to the
    experienced accuracy than the hourly figures are.
    """
    frame = pd.DataFrame({
        "day": np.ceil(dataset["horizon_h"].to_numpy(dtype=float) / 24.0).astype(int),
        "y_true": dataset[truth_col].to_numpy(dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    rows = []
    for day, group in frame.groupby("day"):
        row = {"forecast_day": int(day)}
        row.update(regression_metrics(group["y_true"], group["y_pred"]))
        row.update(operational_metrics(group["y_true"], group["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("forecast_day").reset_index(drop=True)


def summarise(metrics: dict, digits: int = 3) -> str:
    """Compact one-line rendering for logs."""
    keys = ("rmse", "mae", "r2", "skill_vs_persistence", "category_accuracy")
    parts = [f"{k}={metrics[k]:.{digits}f}" for k in keys
             if k in metrics and isinstance(metrics[k], (int, float))
             and np.isfinite(metrics[k])]
    return "  ".join(parts)
