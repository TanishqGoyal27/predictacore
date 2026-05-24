"""
model_manager.py
----------------
Orchestrates the full ML pipeline: training, evaluation, saving, and loading.

This module is the single place where all model artefacts are created or
restored from disk. Flask routes never touch training logic directly.

Training sequence (mirrors the Honeywell notebook):
    1. Generate synthetic dataset (ai4i2020 distribution)
    2. Train/test split (80/20, stratified)
    3. SMOTE oversampling on the training split
    4. Fit FailurePredictor (scales + trains MLP)
    5. Fit AnomalyDetector (trains autoencoder on normal-class samples)
    6. Persist all artefacts to models/
"""

import json
import threading
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.preprocessing import generate_dataset
from src.predictor import FailurePredictor
from src.anomaly_detector import AnomalyDetector

warnings.filterwarnings("ignore")

# Default directory for persisted artefacts (relative to project root)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelManager:
    """
    Manages the lifecycle of all ML artefacts.

    Attributes
    ----------
    predictor        : Trained FailurePredictor (MLP + scaler).
    anomaly_detector : Trained AnomalyDetector (autoencoder + threshold).
    is_ready         : True once models are loaded or trained.
    status           : Human-readable status string for the /api/status endpoint.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir        = models_dir
        self.predictor:         Optional[FailurePredictor] = None
        self.anomaly_detector:  Optional[AnomalyDetector]  = None
        self.is_ready:          bool = False
        self.status:            str  = "idle"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_or_train(self) -> None:
        """Load saved models from disk, or train from scratch if absent."""
        if self._all_artefacts_exist():
            try:
                self._load()
                return
            except Exception as e:
                print(f"[WARN] Could not load saved models ({e}) — retraining.")

        self._train()

    def retrain_async(self) -> None:
        """Trigger a background retraining thread (called by /api/train)."""
        t = threading.Thread(target=self._train, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_artefacts_exist(self) -> bool:
        for name in ["predictor.pkl", "anomaly_detector.pkl"]:
            if not (self.models_dir / name).exists():
                return False
        return True

    def _load(self) -> None:
        print("[LOAD] Loading saved models from disk…")
        self.predictor        = FailurePredictor.load(self.models_dir / "predictor.pkl")
        self.anomaly_detector = AnomalyDetector.load(self.models_dir / "anomaly_detector.pkl")
        self.is_ready = True
        self.status   = "ready"
        print("[LOAD] ✓ Models loaded.")

    def _train(self) -> None:
        with self._lock:
            self.status   = "training"
            self.is_ready = False

        try:
            print("[TRAIN] Generating synthetic dataset (n=10,000)…")
            X, y = generate_dataset(10_000)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            print("[TRAIN] Applying SMOTE oversampling…")
            smote         = SMOTE(random_state=42)
            X_res, y_res  = smote.fit_resample(X_train, y_train)

            print("[TRAIN] Training MLP classifier (64→32→sigmoid)…")
            predictor = FailurePredictor()
            predictor.fit(X_res, y_res)

            recall = predictor.evaluate_recall(X_test, y_test)
            print(f"[TRAIN] Test Recall @ threshold {predictor.threshold}: {recall:.3f}")

            print("[TRAIN] Training autoencoder on normal-class samples…")
            X_train_scaled = predictor.transform(X_res)
            X_normal_scaled = X_train_scaled[y_res == 0]

            detector = AnomalyDetector(
                percentile=99.0,
                input_dim=X_train_scaled.shape[1],
                hidden1=32,
                bottleneck=16,
                lr=0.005,
                epochs=60,
                batch_size=128,
            )
            detector.fit(X_normal_scaled)

            # Persist
            self.models_dir.mkdir(parents=True, exist_ok=True)
            predictor.save(self.models_dir / "predictor.pkl")
            detector.save(self.models_dir / "anomaly_detector.pkl")

            self.predictor        = predictor
            self.anomaly_detector = detector
            self.is_ready = True
            self.status   = "ready"
            print("[TRAIN] ✓ All models saved. System ready.")

        except Exception as e:
            self.status = "error"
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Training failed: {e}") from e


# Single application-wide instance (import this in app.py and api_routes.py)
manager = ModelManager()
