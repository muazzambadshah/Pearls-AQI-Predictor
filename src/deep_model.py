"""
Deep-learning candidate.

A PyTorch MLP wrapped in the scikit-learn estimator interface, so it drops
straight into the same zoo, backtest and registry as everything else with no
special-casing anywhere upstream.

Why an MLP and not an LSTM
--------------------------
A recurrent encoder over the raw hourly sequence is the more obvious "deep
learning for time series" answer, and it was the first thing tried. It is not
the right fit here, for a concrete reason: the temporal structure is already
encoded in the feature matrix as explicit lags, rolling statistics and
origin-time deltas. An LSTM would have to rediscover from raw sequence exactly
what those columns already state directly, on ~29k hourly observations - far
less data than recurrent models need to earn their extra capacity. It also
forces a second, 3-D data path through the entire pipeline for the sake of one
candidate.

The MLP keeps the tabular contract, trains in seconds on CPU, and sits on the
same anchored-residual target as the boosted models. `report.md` records how it
actually placed against them rather than assuming either way.

Kept behind a lazy import so PyTorch never becomes a hard dependency: if torch
is absent the zoo simply omits this candidate.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """
    Feed-forward network with standardised inputs, dropout, early stopping and
    a plateau LR schedule.

    Deliberately mirrors what the gradient-boosted models get for free:
    validation-based stopping so it neither underfits nor memorises, and input
    scaling because unlike a tree an MLP is entirely at the mercy of feature
    magnitudes - AQI counts in the hundreds while `hour_sin` sits in [-1, 1].
    """

    def __init__(self,
                 input_dim: int | None = None,
                 hidden: tuple = (256, 128, 64),
                 dropout: float = 0.15,
                 epochs: int = 40,
                 batch_size: int = 512,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 validation_fraction: float = 0.1,
                 patience: int = 6,
                 random_state: int = 42,
                 verbose: bool = False):
        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    # -- internals ---------------------------------------------------------
    def _build(self, torch, nn, input_dim: int):
        layers, prev = [], input_dim
        for width in self.hidden:
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width),
                       nn.ReLU(), nn.Dropout(self.dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _as_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = self._as_array(X)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        # Standardise on training statistics only, and guard zero-variance
        # columns so a constant feature cannot produce inf.
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.scale_ = np.nanstd(X_arr, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        X_arr = np.nan_to_num((X_arr - self.mean_) / self.scale_, nan=0.0,
                              posinf=0.0, neginf=0.0)

        # Chronological validation split - the rows arrive in time order, so
        # taking the tail keeps early stopping honest rather than optimistic.
        n_val = max(1, int(len(X_arr) * self.validation_fraction))
        X_train, y_train = X_arr[:-n_val], y_arr[:-n_val]
        X_val, y_val = X_arr[-n_val:], y_arr[-n_val:]

        device = torch.device("cpu")
        model = self._build(torch, nn, X_arr.shape[1]).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=2
        )
        # Huber: the AQI spikes that matter most are exactly the points a
        # squared loss would let dominate every gradient step.
        criterion = nn.HuberLoss(delta=10.0)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=len(train_ds) > self.batch_size
        )
        X_val_t = torch.from_numpy(X_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)

        best_loss, best_state, stale = float("inf"), None, 0

        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = criterion(model(xb.to(device)).squeeze(-1), yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(X_val_t).squeeze(-1), y_val_t))
            scheduler.step(val_loss)

            if val_loss < best_loss - 1e-4:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    if self.verbose:
                        logger.info("Early stop at epoch %d (val %.4f)", epoch + 1, best_loss)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        self.model_ = model
        self.n_features_in_ = X_arr.shape[1]
        self.best_val_loss_ = best_loss
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X):
        import torch

        X_arr = self._as_array(X)
        X_arr = np.nan_to_num((X_arr - self.mean_) / self.scale_, nan=0.0,
                              posinf=0.0, neginf=0.0)
        with torch.no_grad():
            out = self.model_(torch.from_numpy(X_arr)).squeeze(-1).numpy()
        return np.asarray(out, dtype=float)

    def __getstate__(self):
        """
        Move tensors to CPU before pickling so a model trained anywhere can be
        loaded by the dashboard and API without a GPU present.
        """
        state = self.__dict__.copy()
        model = state.get("model_")
        if model is not None:
            state["model_"] = model.cpu()
        return state


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
