"""Tests for the NumpyAutoencoder and AnomalyDetector."""

import numpy as np
import pytest

from src.anomaly_detector import NumpyAutoencoder, AnomalyDetector


def make_normal_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 10)).astype(np.float64)


class TestNumpyAutoencoder:
    def test_output_shape_matches_input(self):
        ae = NumpyAutoencoder(input_dim=10, epochs=2)
        X  = make_normal_data(50)
        ae.fit(X)
        out = ae.reconstruct(X)
        assert out.shape == X.shape

    def test_reconstruction_error_shape(self):
        ae = NumpyAutoencoder(input_dim=10, epochs=2)
        X  = make_normal_data(50)
        ae.fit(X)
        err = ae.reconstruction_error(X)
        assert err.shape == (50,)

    def test_reconstruction_error_non_negative(self):
        ae = NumpyAutoencoder(input_dim=10, epochs=2)
        X  = make_normal_data(50)
        ae.fit(X)
        assert (ae.reconstruction_error(X) >= 0).all()

    def test_training_reduces_loss(self):
        ae     = NumpyAutoencoder(input_dim=10, epochs=5, lr=0.01)
        X      = make_normal_data(200)
        before = float(np.mean((X - ae.reconstruct(X)) ** 2))
        ae.fit(X)
        after  = float(np.mean((X - ae.reconstruct(X)) ** 2))
        assert after < before


class TestAnomalyDetector:
    def test_fit_sets_threshold(self):
        X_normal = make_normal_data(200)
        det = AnomalyDetector(percentile=99.0,
                              input_dim=10, hidden1=16, bottleneck=8, epochs=5)
        det.fit(X_normal)
        assert det.threshold > 0.0

    def test_normal_samples_mostly_not_anomalous(self):
        rng      = np.random.default_rng(0)
        X_normal = rng.normal(0, 1, (300, 10)).astype(np.float64)
        det = AnomalyDetector(percentile=99.0,
                              input_dim=10, hidden1=16, bottleneck=8, epochs=10)
        det.fit(X_normal)
        flags = det.is_anomaly(X_normal)
        # At the 99th pct, ~1% of normals should be flagged
        assert flags.mean() < 0.05

    def test_extreme_outlier_flagged(self):
        rng      = np.random.default_rng(0)
        X_normal = rng.normal(0, 1, (300, 10)).astype(np.float64)
        det = AnomalyDetector(percentile=99.0,
                              input_dim=10, hidden1=16, bottleneck=8, epochs=10)
        det.fit(X_normal)

        outlier = np.full((1, 10), 50.0)  # wildly out of distribution
        assert det.is_anomaly(outlier)[0] is True or det.score(outlier)[0] > det.threshold
