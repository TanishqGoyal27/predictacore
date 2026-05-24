"""
anomaly_detector.py
-------------------
Custom NumPy autoencoder for unsupervised anomaly detection.

Architecture:  input(10) → 32 → 16 → 32 → output(10)
Training:      on NORMAL-class samples only (mirrors notebook logic)
Detection:     samples whose reconstruction MSE exceeds the 99th-percentile
               threshold (computed on the normal training set) are flagged
               as anomalies.

No external deep-learning libraries required — only NumPy.
"""

import numpy as np
import pickle
from pathlib import Path


class NumpyAutoencoder:
    """
    Lightweight fully-connected autoencoder trained with mini-batch SGD.

    Mirrors the Keras autoencoder from the Honeywell notebook:
        Encoder: Dense(32, relu) → Dense(16, relu)
        Decoder: Dense(32, relu) → Dense(input_dim, linear)

    Parameters
    ----------
    input_dim  : Number of input features (default 10).
    hidden1    : Size of the first encoder / last decoder layer (default 32).
    bottleneck : Size of the bottleneck (latent) layer (default 16).
    lr         : Learning rate (default 0.005).
    epochs     : Training epochs (default 60).
    batch_size : Mini-batch size (default 128).
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden1: int = 32,
        bottleneck: int = 16,
        lr: float = 0.005,
        epochs: int = 60,
        batch_size: int = 128,
    ):
        self.input_dim  = input_dim
        self.hidden1    = hidden1
        self.bottleneck = bottleneck
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (He uniform)
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        rng = np.random.default_rng(42)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))

        self.W1 = he(self.input_dim,  self.hidden1);    self.b1 = np.zeros(self.hidden1)
        self.W2 = he(self.hidden1,    self.bottleneck); self.b2 = np.zeros(self.bottleneck)
        self.W3 = he(self.bottleneck, self.hidden1);    self.b3 = np.zeros(self.hidden1)
        self.W4 = he(self.hidden1,    self.input_dim);  self.b4 = np.zeros(self.input_dim)

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _drelu(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)

    # ------------------------------------------------------------------
    # Forward pass (stores activations needed by backward)
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> np.ndarray:
        self._z1 = X @ self.W1 + self.b1;        self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2; self._a2 = self._relu(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3; self._a3 = self._relu(self._z3)
        self._z4 = self._a3 @ self.W4 + self.b4  # linear output layer
        return self._z4

    # ------------------------------------------------------------------
    # Backward pass (MSE gradient, gradient clipping for stability)
    # ------------------------------------------------------------------

    def _backward(self, X: np.ndarray, out: np.ndarray) -> None:
        n = float(X.shape[0])

        d4  = 2.0 * (out - X) / n
        dW4 = self._a3.T @ d4;          db4 = d4.sum(0)

        d3  = (d4 @ self.W4.T) * self._drelu(self._z3)
        dW3 = self._a2.T @ d3;          db3 = d3.sum(0)

        d2  = (d3 @ self.W3.T) * self._drelu(self._z2)
        dW2 = self._a1.T @ d2;          db2 = d2.sum(0)

        d1  = (d2 @ self.W2.T) * self._drelu(self._z1)
        dW1 = X.T @ d1;                 db1 = d1.sum(0)

        for dW, W, db, b in [
            (dW1, self.W1, db1, self.b1),
            (dW2, self.W2, db2, self.b2),
            (dW3, self.W3, db3, self.b3),
            (dW4, self.W4, db4, self.b4),
        ]:
            np.clip(dW, -5.0, 5.0, out=dW)
            np.clip(db, -5.0, 5.0, out=db)
            W -= self.lr * dW
            b -= self.lr * db

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "NumpyAutoencoder":
        """Train the autoencoder on X (should contain only NORMAL samples)."""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(0)

        for epoch in range(self.epochs):
            idx = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                batch = X[idx[start : start + self.batch_size]]
                out   = self._forward(batch)
                self._backward(batch, out)

            if (epoch + 1) % 20 == 0:
                loss = float(np.mean((X - self._forward(X)) ** 2))
                print(f"  AE Epoch {epoch + 1:3d}/{self.epochs}  MSE: {loss:.5f}")

        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Return the reconstructed output for each input row."""
        return self._forward(np.asarray(X, dtype=np.float64))

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample mean squared reconstruction error (shape: (n,))."""
        X     = np.asarray(X, dtype=np.float64)
        recon = self.reconstruct(X)
        return np.mean((X - recon) ** 2, axis=1)


class AnomalyDetector:
    """
    Thin wrapper around NumpyAutoencoder that manages threshold computation
    and anomaly flagging.

    Usage
    -----
    detector = AnomalyDetector()
    detector.fit(X_normal_scaled)           # train + set threshold
    flag = detector.is_anomaly(x_scaled)    # True / False
    """

    def __init__(self, percentile: float = 99.0, **ae_kwargs):
        self.percentile = percentile
        self.threshold: float = 0.0
        self.autoencoder = NumpyAutoencoder(**ae_kwargs)

    def fit(self, X_normal: np.ndarray) -> "AnomalyDetector":
        """Train on normal-class samples and set the anomaly threshold."""
        self.autoencoder.fit(X_normal)
        errors = self.autoencoder.reconstruction_error(X_normal)
        self.threshold = float(np.percentile(errors, self.percentile))
        print(f"[AE] Anomaly threshold ({self.percentile}th pct): {self.threshold:.6f}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return raw reconstruction error for each row."""
        return self.autoencoder.reconstruction_error(X)

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Return boolean array — True where reconstruction error > threshold."""
        return self.score(X) > self.threshold

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        with open(path, "rb") as f:
            return pickle.load(f)
