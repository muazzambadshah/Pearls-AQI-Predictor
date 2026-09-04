"""
Training Pipeline - runs daily.

  1. Read raw observations from the Feature Store.
  2. Build the direct multi-horizon supervised dataset.
  3. Walk-forward backtest every candidate model *and* every baseline under
     identical folds.
  4. Pick the winner on mean backtest RMSE, refit it on the full history, and
     promote it to production in the Model Registry.
  5. Persist the per-horizon accuracy curve that the report and dashboard read.

Selecting on the backtest average rather than on a single holdout is the point:
it is the difference between "this model happened to do well on the last three
weeks" and "this model does well repeatedly, on data it had never seen."

Scheduled by .github/workflows/training_pipeline.yml.

Usage
-----
    python -m src.train
    python -m src.train --fast              # smaller ensembles, for CI
    python -m src.train --stride 3          # subsample origins to save memory
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import (backtest, baselines, config, evaluate, feature_engineering,
                 feature_store, model_registry, models)
from src.feature_engineering import split_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fingerprint(raw: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    return {
        "raw_rows": int(len(raw)),
        "raw_start": str(raw.index.min()),
        "raw_end": str(raw.index.max()),
        "dataset_rows": int(len(dataset)),
        "n_origins": int(dataset["origin"].nunique()),
        "n_horizons": int(dataset["horizon_h"].nunique()),
        "n_features": int(len(feature_engineering.feature_columns(dataset))),
    }


def _choose_stride(n_origins: int, n_horizons: int, max_rows: int) -> int:
    """
    Pick an origin stride that keeps the expanded dataset under `max_rows`.

    Every origin fans out into one row per horizon, so three years of hourly
    history at 72 horizons is roughly 1.9M rows. Striding the origins thins that
    without dropping any horizon from the training distribution - each horizon
    stays equally represented, there are simply fewer starting points.
    """
    projected = n_origins * n_horizons
    if projected <= max_rows:
        return 1
    stride = int(np.ceil(projected / max_rows))
    logger.info("Projected %d rows exceeds cap %d - striding origins by %d",
                projected, max_rows, stride)
    return stride


def build_training_dataset(raw: pd.DataFrame, stride: int | None = None):
    """Assemble the supervised table, choosing a stride automatically if needed."""
    n_horizons = len(config.TRAIN_HORIZONS)
    if stride is None:
        stride = _choose_stride(len(raw), n_horizons, config.MAX_TRAIN_ROWS)

    logger.info("Building supervised dataset (stride=%d, horizons=%d)", stride, n_horizons)
    dataset = feature_engineering.make_supervised(
        raw, horizons=config.TRAIN_HORIZONS, origin_stride=stride
    )
    if dataset.empty:
        raise RuntimeError(
            "Supervised dataset is empty. The stored history is probably too "
            "short - run `python -m src.backfill` first."
        )

    logger.info("Dataset: %d rows | %d origins | %d features | %s -> %s",
                len(dataset), dataset["origin"].nunique(),
                len(feature_engineering.feature_columns(dataset)),
                dataset["origin"].min(), dataset["origin"].max())
    return dataset


def _search_blends(predictions: pd.DataFrame,
                   summary: pd.DataFrame,
                   raw_target: pd.Series,
                   fold_cutoffs: dict | None = None,
                   fold_train_rows: dict | None = None,
                   max_members: int = 4):
    """
    Score equal-weighted blends of the top-k candidates, post hoc.

    This costs almost nothing. `compare_models` already collected every model's
    predictions on the *same* test rows, so a blend is an average of columns we
    already have - no refitting, no extra folds. The blends are then scored
    through the identical metric path as the individual models, which keeps the
    comparison honest: a blend appears on the leaderboard on the same terms as
    everything else and only wins if it earns it.

    Returns `(fold_metrics, blend_predictions)` in the same shapes
    `backtest.compare_models` produces, so blends flow through the rest of the
    pipeline - leaderboard, per-horizon curve, promotion - unchanged.
    """
    if predictions is None or predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    ranked = summary[~summary["model"].astype(str).str.startswith("baseline:")]
    ranked = ranked.sort_values("rmse")["model"].tolist()
    if len(ranked) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Align every model's predictions onto one row per test point.
    wide = predictions.pivot_table(
        index=["fold", "origin", "target_time", "horizon_h"],
        columns="model", values="y_pred", aggfunc="first",
    )
    truth = (predictions.groupby(["fold", "origin", "target_time", "horizon_h"])["y"]
             .first().reindex(wide.index))

    rows, pred_frames = [], []
    for k in range(2, min(max_members, len(ranked)) + 1):
        members = [m for m in ranked[:k] if m in wide.columns]
        if len(members) < 2:
            continue

        subset = wide[members].dropna()
        if subset.empty:
            continue

        name = f"blend_top{k}"
        frame = subset.reset_index()[["fold", "origin", "target_time", "horizon_h"]]
        frame["y"] = truth.reindex(subset.index).to_numpy()
        frame["y_pred"] = subset.mean(axis=1).to_numpy()
        frame["model"] = name

        fold_rmse = []
        for fold_id, group in frame.groupby("fold"):
            # Refit climatology at the fold's own cutoff, the same instant the
            # individual models used. Falling back to the first test origin
            # would fit it on one extra hour and make the skill-vs-climatology
            # column subtly incomparable between blends and single models.
            cutoff = (fold_cutoffs or {}).get(fold_id, group["origin"].min())
            climatology = baselines.fit_climatology(raw_target, cutoff)
            base_preds = baselines.compute_all(raw_target, group, climatology=climatology)

            metrics = evaluate.evaluate_predictions(group["y"], group["y_pred"],
                                                    baseline_preds=base_preds)
            # A blend never trains on its own - it averages member models' test
            # predictions - but its members shared the same expanding-window
            # cutoff, so their train_rows count is a truthful, well-defined
            # number for this fold too. Without this, the blend's fold rows
            # carry no train_rows at all; concatenating them with real model
            # folds then averaging in aggregate_folds() produces NaN for every
            # blend's train_rows, which crashes JSON serialization wherever
            # that metrics dict gets returned (e.g. the /model endpoint) the
            # moment a blend is the production model.
            train_rows = (fold_train_rows or {}).get(int(fold_id))
            rows.append({"fold": int(fold_id), "model": name, "cutoff": cutoff,
                         "train_rows": train_rows, "test_rows": int(len(group)),
                         **metrics})
            fold_rmse.append(metrics["rmse"])

            block = group.copy()
            for base_name, values in base_preds.items():
                block[f"base_{base_name}"] = values
            pred_frames.append(block)

        logger.info("  %-12s rmse=%6.2f  (%s)", name,
                    float(np.mean(fold_rmse)), " + ".join(members))

    blend_predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames \
        else pd.DataFrame()
    return pd.DataFrame(rows), blend_predictions


def _blend_members(name: str, summary: pd.DataFrame, zoo: dict) -> list:
    """Rebuild the (name, estimator) list behind a `blend_topK` label."""
    k = int(name.replace("blend_top", ""))
    ranked = summary[~summary["model"].astype(str).str.startswith("baseline:")]
    ranked = ranked[~ranked["model"].astype(str).str.startswith("blend_")]
    ordered = ranked.sort_values("rmse")["model"].tolist()
    return [(m, zoo[m]) for m in ordered[:k] if m in zoo]


def run(fast: bool = False,
        stride: int | None = None,
        n_folds: int | None = None,
        include_deep: bool | None = None,
        include_tf: bool | None = None) -> dict:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    logger.info("=== Training run %s ===", run_id)

    raw = feature_store.read_observations()
    target = config.TARGET_COLUMN
    raw = raw[raw[target].notna()].sort_index()
    logger.info("Loaded %d observations: %s -> %s", len(raw), raw.index.min(), raw.index.max())

    dataset = build_training_dataset(raw, stride=stride)
    raw_target = raw[target].astype(float)
    fingerprint = _fingerprint(raw, dataset)

    # --- Candidates ---------------------------------------------------------
    zoo = models.build_model_zoo(fast=fast)
    include_deep = config.ENABLE_DEEP_MODEL if include_deep is None else include_deep
    if include_deep:
        n_features = len(feature_engineering.feature_columns(dataset))
        deep = models.build_deep_model(input_dim=n_features, fast=fast)
        if deep is not None:
            zoo["deep_mlp_anchored"] = deep

    include_tf = config.ENABLE_TF_MODEL if include_tf is None else include_tf
    if include_tf:
        n_features = len(feature_engineering.feature_columns(dataset))
        tf_model = models.build_tf_model(input_dim=n_features, fast=fast)
        if tf_model is not None:
            zoo["tf_mlp_anchored"] = tf_model
    logger.info("Candidates: %s", ", ".join(zoo))

    # --- Walk-forward evaluation -------------------------------------------
    n_folds = n_folds or config.N_BACKTEST_FOLDS
    logger.info("--- Walk-forward backtest (%d folds) ---", n_folds)

    base_folds = backtest.baseline_summary(dataset, raw_target, n_folds=n_folds)
    if not base_folds.empty:
        base_summary = backtest.aggregate_folds(base_folds)
        for _, row in base_summary.iterrows():
            logger.info("  %-28s rmse=%.3f mae=%.3f r2=%.3f",
                        row["model"], row["rmse"], row["mae"], row["r2"])

    per_fold, summary, predictions = backtest.compare_models(
        zoo, dataset, raw_target, n_folds=n_folds, collect_predictions=True
    )

    # Blends are evaluated on the predictions already collected above, so they
    # join the leaderboard without another pass over the folds.
    model_summary = backtest.aggregate_folds(per_fold)
    fold_cutoffs = per_fold.drop_duplicates("fold").set_index("fold")["cutoff"].to_dict()
    fold_train_rows = per_fold.drop_duplicates("fold").set_index("fold")["train_rows"].to_dict()
    logger.info("--- Evaluating blends ---")
    blend_folds, blend_preds = _search_blends(predictions, model_summary, raw_target,
                                              fold_cutoffs=fold_cutoffs,
                                              fold_train_rows=fold_train_rows)

    frames = [per_fold]
    if not blend_folds.empty:
        frames.append(blend_folds)
        predictions = pd.concat([predictions, blend_preds], ignore_index=True)
    if not base_folds.empty:
        frames.append(base_folds)
    combined = pd.concat(frames, ignore_index=True)
    full_summary = backtest.aggregate_folds(combined)

    logger.info("--- Backtest leaderboard (mean across folds) ---")
    for _, row in full_summary.iterrows():
        logger.info("  %-32s rmse=%6.2f (+/-%.2f)  mae=%6.2f  r2=%6.3f  skill=%+.3f",
                    row["model"], row["rmse"], row.get("rmse_std", float("nan")),
                    row["mae"], row["r2"], row.get("skill_vs_persistence", float("nan")))

    # --- Winner -------------------------------------------------------------
    model_rows = full_summary[~full_summary["model"].str.startswith("baseline:")]
    if model_rows.empty:
        raise RuntimeError("No trained candidate survived the backtest")
    best_name = model_rows.iloc[0]["model"]
    best_metrics = model_rows.iloc[0].to_dict()
    logger.info("Winner: %s (backtest RMSE %.3f)", best_name, best_metrics["rmse"])

    persistence_row = full_summary[full_summary["model"] == "baseline:persistence"]
    if not persistence_row.empty:
        gain = 1 - best_metrics["rmse"] / persistence_row.iloc[0]["rmse"]
        logger.info("Skill vs persistence: %+.1f%% RMSE reduction", gain * 100)

    # --- Per-horizon accuracy curve ----------------------------------------
    horizon_metrics = pd.DataFrame()
    if predictions is not None:
        best_preds = predictions[predictions["model"] == best_name]
        if not best_preds.empty:
            base_cols = {c.replace("base_", ""): best_preds[c].to_numpy()
                         for c in best_preds.columns if c.startswith("base_")}
            horizon_metrics = evaluate.per_horizon_metrics(
                best_preds,
                best_preds["y_pred"].to_numpy(),
                baseline_preds=base_cols,
            )
            horizon_metrics["model"] = best_name
            horizon_metrics.to_parquet(config.HORIZON_METRICS_PATH)

            daily = evaluate.daily_metrics(best_preds, best_preds["y_pred"].to_numpy())
            logger.info("--- Accuracy by forecast day ---")
            for _, row in daily.iterrows():
                logger.info("  Day %d: rmse=%6.2f mae=%6.2f r2=%6.3f  band-acc=%.3f",
                            row["forecast_day"], row["rmse"], row["mae"],
                            row["r2"], row.get("category_accuracy", float("nan")))

        predictions.to_parquet(config.BACKTEST_PATH)

    combined.to_parquet(config.DATA_DIR / "backtest_folds.parquet")
    full_summary.to_parquet(config.DATA_DIR / "model_comparison.parquet")

    # --- Refit the winner on all history and promote ------------------------
    logger.info("Refitting %s on the full history", best_name)
    X_all, y_all, _ = split_feature_matrix(dataset)

    if best_name.startswith("blend_"):
        members = _blend_members(best_name, full_summary, zoo)
        logger.info("Blend members: %s", ", ".join(name for name, _ in members))
        final_model = models.BlendedRegressor(members=members)
    else:
        final_model = zoo[best_name]

    final_model.fit(X_all, y_all)

    feature_names = feature_engineering.feature_columns(dataset)
    model_registry.register(
        final_model, best_name,
        metrics={k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
        feature_names=feature_names,
        run_id=run_id,
        data_fingerprint=fingerprint,
        horizon_metrics=horizon_metrics.to_dict("records") if not horizon_metrics.empty else [],
        extra={"selection": "mean backtest RMSE across walk-forward folds"},
    )
    model_registry.promote(run_id, best_name)
    model_registry.prune(keep=12)

    logger.info("=== Training run %s complete ===", run_id)
    return {
        "run_id": run_id,
        "best_model": best_name,
        "metrics": {k: float(v) for k, v in best_metrics.items()
                    if isinstance(v, (int, float, np.floating))},
        "fingerprint": fingerprint,
        "leaderboard": full_summary.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily AQI training pipeline")
    parser.add_argument("--fast", action="store_true", help="Smaller ensembles (CI)")
    parser.add_argument("--stride", type=int, default=None, help="Origin subsampling stride")
    parser.add_argument("--folds", type=int, default=None, help="Walk-forward fold count")
    parser.add_argument("--deep", action="store_true", help="Include the PyTorch MLP candidate")
    parser.add_argument("--tf", action="store_true", help="Include the TensorFlow/Keras MLP candidate")
    args = parser.parse_args()
    run(fast=args.fast, stride=args.stride, n_folds=args.folds,
        include_deep=True if args.deep else None,
        include_tf=True if args.tf else None)


if __name__ == "__main__":
    main()
