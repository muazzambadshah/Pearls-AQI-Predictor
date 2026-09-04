"""
Walk-forward backtesting.

A single 80/20 chronological split gives one number from one slice of history.
If that slice happened to be a calm July, the model looks excellent; if it
covered a December inversion episode, it looks broken. Neither tells you how the
thing behaves in production, where it is retrained continuously and forecasts
into whatever comes next.

This module instead replays the deployment loop: train on everything up to a
cutoff, forecast the window that follows, roll the cutoff forward, repeat. Every
fold is an honest out-of-sample test, and the spread across folds shows how
stable the result actually is.

The embargo, and why it matters
-------------------------------
Multi-horizon datasets leak across a naive split in a way that is easy to miss.
A training sample with origin `T - 10h` and horizon 72 has its *target* at
`T + 62h`. Splitting on the origin alone would put that sample in the training
set even though its label lies inside the test window - so the model would be
fitted on the very future it is about to be scored against.

The fix is to split on `target_time`, not `origin`:

    train = rows whose target_time <= cutoff
    test  = rows whose origin     >  cutoff

which leaves a natural 72-hour embargo between the two and makes each fold a
faithful replay of what the model could genuinely have known at the cutoff.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import baselines, config, evaluate
from src.feature_engineering import split_feature_matrix

logger = logging.getLogger(__name__)


def make_folds(dataset: pd.DataFrame, n_folds: int = 4, min_train_fraction: float = 0.4):
    """
    Expanding-window fold boundaries.

    Yields `(cutoff, fold_end)` timestamps. The training window always starts at
    the beginning of history and grows - the same way a daily retrain behaves in
    production - rather than sliding a fixed-width window.
    """
    origins = pd.DatetimeIndex(dataset["origin"].unique()).sort_values()
    if len(origins) < 10:
        raise ValueError("Not enough distinct origins to build backtest folds")

    start_idx = int(len(origins) * min_train_fraction)
    remaining = len(origins) - start_idx
    if remaining < n_folds:
        n_folds = max(1, remaining)
    step = remaining // n_folds

    for i in range(n_folds):
        cutoff = origins[start_idx + i * step]
        end_idx = start_idx + (i + 1) * step - 1 if i < n_folds - 1 else len(origins) - 1
        yield cutoff, origins[end_idx]


def split_at(dataset: pd.DataFrame, cutoff, fold_end=None):
    """
    Split one fold with the target-time embargo described in the module docstring.
    """
    cutoff = pd.Timestamp(cutoff)
    train_mask = dataset["target_time"] <= cutoff
    test_mask = dataset["origin"] > cutoff
    if fold_end is not None:
        test_mask &= dataset["origin"] <= pd.Timestamp(fold_end)
    return dataset.loc[train_mask], dataset.loc[test_mask]


def _fit_and_score(model, train_df, test_df, raw_target, cutoff):
    """Fit one candidate on a fold and score it against every baseline."""
    from sklearn.base import clone

    X_train, y_train, _ = split_feature_matrix(train_df)
    X_test, y_test, _ = split_feature_matrix(test_df)

    fitted = clone(model)
    fitted.fit(X_train, y_train)
    preds = np.asarray(fitted.predict(X_test), dtype=float)

    climatology = baselines.fit_climatology(raw_target, cutoff)
    base_preds = baselines.compute_all(raw_target, test_df, climatology=climatology)

    metrics = evaluate.evaluate_predictions(y_test, preds, baseline_preds=base_preds)
    return fitted, preds, metrics, base_preds


def backtest_model(model,
                   dataset: pd.DataFrame,
                   raw_target: pd.Series,
                   n_folds: int | None = None,
                   model_name: str = "model",
                   collect_predictions: bool = False):
    """
    Replay `model` across expanding-window folds.

    Returns `(fold_metrics_df, predictions_df_or_None)`.
    """
    n_folds = n_folds or config.N_BACKTEST_FOLDS
    fold_rows = []
    prediction_frames = []

    for fold_idx, (cutoff, fold_end) in enumerate(make_folds(dataset, n_folds), start=1):
        train_df, test_df = split_at(dataset, cutoff, fold_end)
        if train_df.empty or test_df.empty:
            logger.warning("Fold %d at %s is empty - skipping", fold_idx, cutoff)
            continue

        _, preds, metrics, base_preds = _fit_and_score(
            model, train_df, test_df, raw_target, cutoff
        )

        row = {
            "fold": fold_idx,
            "model": model_name,
            "cutoff": cutoff,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            **metrics,
        }
        fold_rows.append(row)
        logger.info("  fold %d/%d @ %s | %s", fold_idx, n_folds,
                    pd.Timestamp(cutoff).date(), evaluate.summarise(metrics))

        if collect_predictions:
            frame = test_df[["origin", "target_time", "horizon_h", "y"]].copy()
            frame["y_pred"] = preds
            frame["fold"] = fold_idx
            frame["model"] = model_name
            for name, values in base_preds.items():
                frame[f"base_{name}"] = values
            prediction_frames.append(frame)

    if not fold_rows:
        raise RuntimeError(f"Backtest for '{model_name}' produced no usable folds")

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else None
    return fold_metrics, predictions


def aggregate_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-fold scores into one row per model.

    Carries the standard deviation of RMSE alongside the mean: two models with
    the same average can differ sharply in how dependably they achieve it, and
    the steadier one is usually the better thing to deploy.
    """
    numeric = [c for c in fold_metrics.columns
               if c not in ("fold", "model", "cutoff")
               and pd.api.types.is_numeric_dtype(fold_metrics[c])]

    grouped = fold_metrics.groupby("model")[numeric].mean().reset_index()
    grouped["rmse_std"] = fold_metrics.groupby("model")["rmse"].std().values
    grouped["n_folds"] = fold_metrics.groupby("model").size().values
    return grouped.sort_values("rmse").reset_index(drop=True)


def compare_models(zoo: dict,
                   dataset: pd.DataFrame,
                   raw_target: pd.Series,
                   n_folds: int | None = None,
                   collect_predictions: bool = False):
    """
    Backtest every candidate in `zoo` under identical folds.

    Returns `(per_fold_df, summary_df, predictions_df_or_None)`.
    """
    all_folds, all_preds = [], []

    for name, model in zoo.items():
        logger.info("Backtesting %s", name)
        try:
            folds, preds = backtest_model(
                model, dataset, raw_target, n_folds=n_folds,
                model_name=name, collect_predictions=collect_predictions,
            )
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not stop the run
            logger.error("  %s failed: %s", name, exc)
            continue

        all_folds.append(folds)
        if preds is not None:
            all_preds.append(preds)

    if not all_folds:
        raise RuntimeError("Every candidate failed during backtesting")

    per_fold = pd.concat(all_folds, ignore_index=True)
    summary = aggregate_folds(per_fold)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else None
    return per_fold, summary, predictions


def baseline_summary(dataset: pd.DataFrame,
                     raw_target: pd.Series,
                     n_folds: int | None = None) -> pd.DataFrame:
    """
    Score the baselines through the identical fold machinery.

    Running them on the same test rows as the models is what makes the skill
    scores meaningful rather than an apples-to-oranges comparison.
    """
    n_folds = n_folds or config.N_BACKTEST_FOLDS
    rows = []

    for fold_idx, (cutoff, fold_end) in enumerate(make_folds(dataset, n_folds), start=1):
        _, test_df = split_at(dataset, cutoff, fold_end)
        if test_df.empty:
            continue

        climatology = baselines.fit_climatology(raw_target, cutoff)
        base_preds = baselines.compute_all(raw_target, test_df, climatology=climatology)

        for name, preds in base_preds.items():
            metrics = evaluate.regression_metrics(test_df["y"], preds)
            metrics.update(evaluate.operational_metrics(test_df["y"], preds))
            rows.append({"fold": fold_idx, "model": f"baseline:{name}",
                         "cutoff": cutoff, "test_rows": int(len(test_df)), **metrics})

    return pd.DataFrame(rows)
