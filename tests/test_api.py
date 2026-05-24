"""
Smoke tests for Flask API endpoints.

These tests use a mock model manager so they do not require a trained
model to be present on disk — keeping CI fast and self-contained.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app import create_app


# ─── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─── /api/status ──────────────────────────────────────────────────────────────

class TestStatus:
    def test_returns_200(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200

    def test_has_expected_keys(self, client):
        data = json.loads(client.get("/api/status").data)
        assert "status" in data
        assert "ready" in data


# ─── /api/predict — unhappy paths (no model needed) ───────────────────────────

class TestPredictValidation:
    def test_returns_503_when_not_ready(self, client):
        r = client.post(
            "/api/predict",
            data=json.dumps({"type": "M", "air_temp": 300, "proc_temp": 310,
                             "rpm": 1538, "torque": 40, "tool_wear": 100}),
            content_type="application/json",
        )
        # Manager is not ready in a fresh test context
        assert r.status_code == 503

    def test_returns_400_on_missing_field(self, client):
        # Patch manager to appear ready
        with patch("src.api_routes.manager") as mock_mgr:
            mock_mgr.is_ready = True
            r = client.post(
                "/api/predict",
                data=json.dumps({"type": "M"}),
                content_type="application/json",
            )
        assert r.status_code == 400

    def test_returns_400_on_invalid_type(self, client):
        with patch("src.api_routes.manager") as mock_mgr:
            mock_mgr.is_ready = True
            payload = {"type": "Z", "air_temp": 300, "proc_temp": 310,
                       "rpm": 1538, "torque": 40, "tool_wear": 100}
            r = client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
        assert r.status_code == 400

    def test_returns_400_on_out_of_range(self, client):
        with patch("src.api_routes.manager") as mock_mgr:
            mock_mgr.is_ready = True
            payload = {"type": "M", "air_temp": 999, "proc_temp": 310,
                       "rpm": 1538, "torque": 40, "tool_wear": 100}
            r = client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
        assert r.status_code == 400


# ─── /api/predict — happy path (mocked pipeline) ──────────────────────────────

class TestPredictHappyPath:
    VALID_PAYLOAD = {
        "type": "M", "air_temp": 300.5, "proc_temp": 310.5,
        "rpm": 1538, "torque": 40.0, "tool_wear": 180,
    }

    def _mock_manager(self):
        """Build a mock manager that returns plausible model outputs."""
        mock_pred = MagicMock()
        mock_pred.threshold = 0.3
        mock_pred.transform.return_value = np.zeros((1, 10))
        mock_pred.predict_proba_positive.return_value = np.array([0.85])

        mock_ae = MagicMock()
        mock_ae.score.return_value = np.array([0.07])
        mock_ae.threshold = 0.05

        mock_mgr = MagicMock()
        mock_mgr.is_ready = True
        mock_mgr.predictor = mock_pred
        mock_mgr.anomaly_detector = mock_ae
        return mock_mgr

    def test_returns_200(self, client):
        with patch("src.api_routes.manager", self._mock_manager()):
            r = client.post(
                "/api/predict",
                data=json.dumps(self.VALID_PAYLOAD),
                content_type="application/json",
            )
        assert r.status_code == 200

    def test_response_has_required_keys(self, client):
        with patch("src.api_routes.manager", self._mock_manager()):
            r = client.post(
                "/api/predict",
                data=json.dumps(self.VALID_PAYLOAD),
                content_type="application/json",
            )
        data = json.loads(r.data)
        for key in ["prediction", "probability", "risk_level", "confidence",
                    "anomaly", "final_score", "root_causes", "action"]:
            assert key in data, f"Missing key: {key}"

    def test_prediction_is_valid_label(self, client):
        with patch("src.api_routes.manager", self._mock_manager()):
            r = client.post(
                "/api/predict",
                data=json.dumps(self.VALID_PAYLOAD),
                content_type="application/json",
            )
        data = json.loads(r.data)
        assert data["prediction"] in ("FAILURE", "NORMAL")
