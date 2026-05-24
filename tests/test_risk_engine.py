"""Tests for risk scoring, confidence, and maintenance action logic."""

import pytest

from src.risk_engine import (
    compute_risk_score,
    risk_level,
    confidence_level,
    maintenance_action,
)


class TestComputeRiskScore:
    def test_high_prob_high_mse_gives_high_score(self):
        score = compute_risk_score(prob=0.9, anomaly_mse=0.1, anomaly_threshold=0.05)
        assert score > 0.7

    def test_low_prob_low_mse_gives_low_score(self):
        score = compute_risk_score(prob=0.05, anomaly_mse=0.001, anomaly_threshold=0.05)
        assert score < 0.4

    def test_score_bounded_0_to_1(self):
        # Even with very high MSE, score should not exceed 1
        score = compute_risk_score(prob=1.0, anomaly_mse=999.0, anomaly_threshold=0.01)
        assert 0.0 <= score <= 1.0

    def test_formula_correctness(self):
        prob, mse, thr = 0.6, 0.04, 0.05
        expected = 0.7 * 0.6 + 0.3 * (0.04 / 0.05)
        assert compute_risk_score(prob, mse, thr) == pytest.approx(expected)


class TestRiskLevel:
    def test_high(self):
        assert risk_level(0.8) == "HIGH"

    def test_medium(self):
        assert risk_level(0.55) == "MEDIUM"

    def test_low(self):
        assert risk_level(0.2) == "LOW"

    def test_boundary_high(self):
        assert risk_level(0.70) == "MEDIUM"   # not strictly > 0.70

    def test_boundary_medium(self):
        assert risk_level(0.40) == "LOW"      # not strictly > 0.40


class TestConfidenceLevel:
    def test_uncertain(self):
        assert confidence_level(0.5) == "UNCERTAIN"
        assert confidence_level(0.45) == "UNCERTAIN"

    def test_high_confidence_upper(self):
        assert confidence_level(0.9) == "HIGH CONFIDENCE"

    def test_high_confidence_lower(self):
        assert confidence_level(0.1) == "HIGH CONFIDENCE"

    def test_moderate(self):
        assert confidence_level(0.65) == "MODERATE"


class TestMaintenanceAction:
    def test_wear_rate_failure(self):
        action = maintenance_action(pred=1, is_anomaly=False, top_feature="Wear_rate")
        assert "wear" in action.lower()

    def test_torque_failure(self):
        action = maintenance_action(pred=1, is_anomaly=False, top_feature="Torque [Nm]")
        assert "torque" in action.lower() or "motor" in action.lower()

    def test_temperature_failure(self):
        action = maintenance_action(pred=1, is_anomaly=False, top_feature="Temp_diff")
        assert "cooling" in action.lower() or "thermal" in action.lower()

    def test_normal_no_anomaly(self):
        action = maintenance_action(pred=0, is_anomaly=False, top_feature="Type")
        assert "normal" in action.lower()

    def test_normal_with_anomaly(self):
        action = maintenance_action(pred=0, is_anomaly=True, top_feature="Type")
        assert "anomaly" in action.lower() or "monitor" in action.lower()
