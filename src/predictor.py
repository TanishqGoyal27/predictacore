"""
predictor.py
------------
MLP classifier: training, persistence, and probability inference.

Architecture mirrors the Keras model from the Honeywell notebook:
    Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)

Implemented with scikit-learn MLPClassifier, which matches this
architecture when hidden_layer_sizes=(64, 32) and activation='relu'.

Key design choices:
- Prediction threshold set to 0.30 (recall-first strategy from notebook)
- Trained on SMOTE-resampled data to handle the heavy class imbalance
  (~3% failure rate in the ai4i2020 dataset)
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Recall-first threshold: prefer catching failures over precision
PREDICTION_THRESHOLD = 0.30


class FailurePredictor:
    """
    Wraps scikit-learn MLPClassifier with fit / predict_proba / save / load.

    Parameters
    ----------
    threshold : float
        Probability cutoff for classifying a sample as a failure.
        Default 0.30 (recall-first, mirroring the notebook).
    """

    def __init__(self, threshold: float = PREDICTION_THRESHOLD):
        self.threshold = threshold
        self.scaler: StandardScaler = StandardScaler()
        self.model: MLPClassifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            verbose=False,
        )

    def fit(self, X_resampled: np.ndarray, y_resampled: np.ndarray) -> "FailurePredictor":
        """
        Fit the scaler on X_resampled, then train the MLP.

        X_resampled should already be SMOTE-oversampled training data.
        """
        X_scaled = self.scaler.fit_transform(X_resampled).astype(np.float64)
        self.model.fit(X_scaled, y_resampled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler to X."""
        return self.scaler.transform(X).astype(np.float64)

    def predict_proba_positive(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return probability of the positive (failure) class."""
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return binary predictions using self.threshold."""
        return (self.predict_proba_positive(X_scaled) > self.threshold).astype(int)

    def evaluate_recall(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute recall on the test set using the current threshold."""
        X_scaled = self.transform(X_test)
        preds    = self.predict(X_scaled)
        tp       = int(np.sum((preds == 1) & (y_test == 1)))
        fn       = int(np.sum((preds == 0) & (y_test == 1)))
        return tp / max(tp + fn, 1)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "FailurePredictor":
        with open(path, "rb") as f:
            return pickle.load(f)
