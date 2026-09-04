"""
TensorFlow/Keras deep-learning candidate.

Mirrors `deep_model.TorchMLPRegressor` feature-for-feature - same architecture
shape, same standardisation, same Huber loss (the AQI spikes that matter most
are exactly what a squared loss would let dominate every gradient step), same
chronological validation split and early stopping - so the backtest compares
two genuinely equivalent MLPs on the framework axis alone, not on hyper-
parameter choices that happen to differ between them.

Kept behind a lazy import so TensorFlow never becomes a hard dependency: if
`tensorflow` is absent the zoo simply omits this candidate, the same pattern
`deep_model.py` already uses for PyTorch.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class KerasMLPRegressor(BaseEstimator, RegressorMixin):
    """Feed-forward network (TensorFlow/Keras) with standardised inputs,
    dropout, early stopping and a plateau LR schedule."""

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

    @staticmethod
    def _as_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def _build(self, keras, layers, input_dim: int):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        for width in self.hidden:
            model.add(layers.Dense(
                width, kernel_regularizer=self._regularizer(keras)))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1))
        return model

    def _regularizer(self, keras):
        return keras.regularizers.l2(self.weight_decay)

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = self._as_array(X)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        # Standardise on training statistics only, guarding zero-variance
        # columns so a constant feature cannot produce inf.
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.scale_ = np.nanstd(X_arr, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        X_arr = np.nan_to_num((X_arr - self.mean_) / self.scale_, nan=0.0,
                              posinf=0.0, neginf=0.0)

        # Chronological validation split - rows arrive in time order, so the
        # tail is held out to keep early stopping honest rather than optimistic.
        n_val = max(1, int(len(X_arr) * self.validation_fraction))
        X_train, y_train = X_arr[:-n_val], y_arr[:-n_val]
        X_val, y_val = X_arr[-n_val:], y_arr[-n_val:]

        model = self._build(keras, layers, X_arr.shape[1])
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay),
            loss=keras.losses.Huber(delta=10.0),
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2),
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0,
        )

        val_loss = model.evaluate(X_val, y_val, verbose=0)
        self.model_ = model
        self.n_features_in_ = X_arr.shape[1]
        self.best_val_loss_ = float(val_loss)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X):
        X_arr = self._as_array(X)
        X_arr = np.nan_to_num((X_arr - self.mean_) / self.scale_, nan=0.0,
                              posinf=0.0, neginf=0.0)
        preds = self.model_.predict(X_arr, verbose=0)
        return preds.reshape(-1)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_")
