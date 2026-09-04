"""
The candidate model zoo.

Spans the range the brief asks for - classical statistical regressors, tree
ensembles, gradient boosting, and (optionally) a deep sequence model - behind
one uniform scikit-learn-style interface so the training pipeline can treat them
interchangeably and let the backtest pick a winner.

Anchored targets
----------------
`AnchoredRegressor` learns the *change* from the last observed value rather than
the AQI level itself:

        y_delta = aqi[t + h] - aqi[t]
        prediction = aqi[t] + model.predict(X)

The argument for it is that tree ensembles cannot extrapolate - a tree trained
on levels can never predict an AQI above the highest value in its training data,
which is exactly the regime that matters during a severe smog episode - and that
the delta distribution is more stationary than the level distribution across
Lahore's very different summer and winter baselines.

Measured across the walk-forward backtest, the framing turns out to be close to
a wash, and which way it falls depends on the model:

    ridge                 29.50  |  ridge_anchored                 29.50
    lightgbm              28.54  |  lightgbm_anchored              28.63
    hist_gradient_boosting 28.92 |  hist_gradient_boosting_anchored 28.72

Every one of those gaps is an order of magnitude smaller than the fold-to-fold
standard deviation (~8-9 RMSE), so the honest reading is that this dataset does
not decide the question rather than that either framing wins. A plausible reason
the effect is so muted: `aqi_at_origin` is already a feature, so a level-target
model can learn the same residual relationship wherever it helps, which leaves
the anchored version with little to add.

The one place it clearly earns its keep is the neural network. `deep_mlp_anchored`
is the best single model in the current run, and an MLP has none of a tree's
built-in tolerance for a shifting target level - it benefits from being handed a
stationary target in a way the tree ensembles do not.

Both framings stay in the zoo precisely because of this: the backtest settles it
with numbers rather than argument. See `reports/report.md`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.feature_engineering import ANCHOR_COLUMN

logger = logging.getLogger(__name__)

# AQI is defined on a bounded scale; predictions outside it are meaningless.
AQI_MIN, AQI_MAX = 0.0, 500.0

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LIGHTGBM = False
    logger.info("LightGBM not installed - skipping boosted candidates.")


class AnchoredRegressor(BaseEstimator, RegressorMixin):
    """
    Fits `base` against the residual from the persistence anchor.

    The anchor column travels inside `X`, so this composes with anything that
    accepts a DataFrame and needs no changes elsewhere in the pipeline.
    """

    def __init__(self, base=None, anchor_column: str = ANCHOR_COLUMN,
                 clip: bool = True):
        self.base = base
        self.anchor_column = anchor_column
        self.clip = clip

    def _anchor(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.anchor_column not in X.columns:
                raise KeyError(f"Anchor column '{self.anchor_column}' missing from X")
            return X[self.anchor_column].to_numpy(dtype=float)
        raise TypeError("AnchoredRegressor requires a pandas DataFrame")

    def fit(self, X, y):
        anchor = self._anchor(X)
        self.base_ = clone(self.base)
        self.base_.fit(X, np.asarray(y, dtype=float) - anchor)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X):
        anchor = self._anchor(X)
        pred = anchor + np.asarray(self.base_.predict(X), dtype=float)
        return np.clip(pred, AQI_MIN, AQI_MAX) if self.clip else pred

    @property
    def feature_importances_(self):
        return getattr(self.base_, "feature_importances_", None)


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """Level-target model whose output is clamped to the valid AQI range."""

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.base_ = clone(self.base)
        self.base_.fit(X, y)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X):
        return np.clip(np.asarray(self.base_.predict(X), dtype=float), AQI_MIN, AQI_MAX)

    @property
    def feature_importances_(self):
        return getattr(self.base_, "feature_importances_", None)


class BlendedRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted average of several fitted candidates.

    Different model families make different mistakes - a boosted tree and a
    random forest disagree on which regions of the feature space are hard - so
    averaging them cancels part of the error that no single one can remove.
    It is the cheapest reliable accuracy gain available here.

    The members are chosen by the backtest rather than picked in advance, and
    the blend is scored through the identical folds as everything else, so it
    only ships if it actually wins. See `train._search_blends`.
    """

    def __init__(self, members=None, weights=None, clip: bool = True):
        self.members = members          # list of (name, estimator)
        self.weights = weights          # None -> equal weighting
        self.clip = clip

    def _normalised_weights(self, n: int) -> np.ndarray:
        if self.weights is None:
            return np.full(n, 1.0 / n)
        weights = np.asarray(self.weights, dtype=float)
        total = weights.sum()
        if total <= 0:
            raise ValueError("Blend weights must sum to a positive number")
        return weights / total

    def fit(self, X, y):
        if not self.members:
            raise ValueError("BlendedRegressor needs at least one member")
        self.fitted_ = [(name, clone(est).fit(X, y)) for name, est in self.members]
        self.weights_ = self._normalised_weights(len(self.fitted_))
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X):
        preds = np.column_stack([est.predict(X) for _, est in self.fitted_])
        out = preds @ self.weights_
        return np.clip(out, AQI_MIN, AQI_MAX) if self.clip else out

    @property
    def member_names(self) -> list[str]:
        return [name for name, _ in (self.members or [])]


def _linear_pipeline(estimator) -> Pipeline:
    """Linear models need imputation and scaling; trees need neither."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _imputed(estimator) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


def build_model_zoo(fast: bool = False) -> dict:
    """
    Every candidate the training pipeline will fit and score.

    `fast=True` shrinks the ensembles so CI and the test suite stay quick while
    still exercising each code path.
    """
    seed = config.RANDOM_SEED
    # The forests are the runtime bottleneck of a full backtest and have never
    # placed near the top, so they are sized to be a fair reference rather than
    # a contender - the brief names Random Forest explicitly, so it stays.
    n_estimators = 120 if fast else 150
    lgbm_rounds = 200 if fast else 900
    # Each forest tree holds its own bootstrap sample, so on a 250k-row dataset
    # an unrestricted forest is the memory ceiling of the whole run. Bagging 50%
    # per tree costs very little accuracy on data this correlated and roughly
    # halves the footprint.
    max_samples = 0.5 if not fast else None

    zoo: dict = {}

    # --- Statistical / linear -------------------------------------------------
    zoo["ridge"] = ClippedRegressor(_linear_pipeline(Ridge(alpha=5.0, random_state=seed)))
    zoo["elastic_net"] = ClippedRegressor(
        _linear_pipeline(ElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=5000, random_state=seed))
    )
    zoo["ridge_anchored"] = AnchoredRegressor(
        _linear_pipeline(Ridge(alpha=5.0, random_state=seed))
    )

    # --- Tree ensembles -------------------------------------------------------
    zoo["random_forest"] = ClippedRegressor(_imputed(RandomForestRegressor(
        n_estimators=n_estimators, max_depth=18, min_samples_leaf=8,
        max_features=0.35, max_samples=max_samples,
        random_state=seed, n_jobs=-1,
    )))
    zoo["extra_trees_anchored"] = AnchoredRegressor(_imputed(ExtraTreesRegressor(
        n_estimators=n_estimators, max_depth=20, min_samples_leaf=8,
        max_features=0.35, bootstrap=True, max_samples=max_samples,
        random_state=seed, n_jobs=-1,
    )))

    # HistGradientBoosting handles NaN natively, so it needs no imputer.
    zoo["hist_gradient_boosting"] = ClippedRegressor(HistGradientBoostingRegressor(
        max_iter=lgbm_rounds, learning_rate=0.06, max_depth=None,
        max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=seed,
    ))
    zoo["hist_gradient_boosting_anchored"] = AnchoredRegressor(HistGradientBoostingRegressor(
        max_iter=lgbm_rounds, learning_rate=0.06, max_depth=None,
        max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=seed,
    ))

    # --- Gradient boosting ----------------------------------------------------
    if HAS_LIGHTGBM:
        def _lgbm(objective="l2"):
            return LGBMRegressor(
                n_estimators=lgbm_rounds, learning_rate=0.05, num_leaves=63,
                min_child_samples=40, subsample=0.85, subsample_freq=1,
                colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                objective=objective, random_state=seed, n_jobs=-1, verbose=-1,
            )

        zoo["lightgbm"] = ClippedRegressor(_lgbm())
        zoo["lightgbm_anchored"] = AnchoredRegressor(_lgbm())

        # A Huber-loss candidate was here, on the reasoning that the extreme
        # smog spikes dominate a squared-error gradient and a robust loss would
        # temper them. It was removed after measurement, on both counts:
        #
        #   Speed - LightGBM's huber objective took over four hours to reach
        #   fold 3 on this dataset, against under three minutes for all four
        #   folds of the identical model under L2. Huber cannot use the fast
        #   Newton leaf-value step that L2 gets, so every leaf falls back to a
        #   far slower path.
        #
        #   Accuracy - in the runs where it did finish, it placed last of every
        #   tree candidate (skill +0.06 against +0.27 for plain LightGBM).
        #   Down-weighting the spikes hurts here, because the spikes are the
        #   episodes the forecast exists to catch.
        #
        # Robust losses are still the right answer for the interval calibration
        # problem described in the report; quantile regression is the route
        # there, not a robust point loss.

    return zoo


def model_requirements_probe(model) -> list[str]:
    """Convenience re-export so tests and callers need only one import."""
    from src.model_registry import model_requirements

    return model_requirements(model)


def build_deep_model(input_dim: int, fast: bool = False):
    """
    Optional deep-learning candidate.

    Prefers PyTorch (reliable wheels on every platform we target) and falls back
    to TensorFlow/Keras if that is what is installed. Returns None when neither
    is available, which is the normal case in CI.
    """
    try:
        from src.deep_model import TorchMLPRegressor
        return AnchoredRegressor(TorchMLPRegressor(
            input_dim=input_dim,
            hidden=(256, 128, 64) if not fast else (64, 32),
            epochs=40 if not fast else 5,
            random_state=config.RANDOM_SEED,
        ))
    except Exception as exc:  # noqa: BLE001
        logger.info("Deep model unavailable (%s) - skipping.", exc)
        return None


def build_tf_model(input_dim: int, fast: bool = False):
    """
    TensorFlow/Keras deep-learning candidate, evaluated alongside the PyTorch
    MLP rather than instead of it - both frameworks the brief names are then
    genuinely present in the leaderboard, not just one behind a fallback path.

    Returns None when `tensorflow` is not installed, so it never becomes a
    hard dependency (same pattern as `build_deep_model`).
    """
    try:
        from src.tf_model import KerasMLPRegressor
        return AnchoredRegressor(KerasMLPRegressor(
            input_dim=input_dim,
            hidden=(256, 128, 64) if not fast else (64, 32),
            epochs=40 if not fast else 5,
            random_state=config.RANDOM_SEED,
        ))
    except Exception as exc:  # noqa: BLE001
        logger.info("TensorFlow model unavailable (%s) - skipping.", exc)
        return None
