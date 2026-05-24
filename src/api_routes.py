"""
api_routes.py
-------------
Flask Blueprint containing all /api/* routes.

Importing as a Blueprint keeps the route definitions separate from the
Flask app factory (app.py), making the codebase easier to test and extend.

Endpoints
---------
GET  /api/status    — Model readiness and training status.
POST /api/predict   — Run the prediction pipeline on a single input.
GET  /api/simulate  — Generate 8 random machine predictions (monitoring demo).
POST /api/train     — Trigger a background model retraining.
"""

import numpy as np
from flask import Blueprint, jsonify, request

from src.model_manager import manager
from src.preprocessing import build_features, validate_input
from src.explainability import get_feature_importance, format_for_chart
from src.risk_engine import compute_risk_score, risk_level, confidence_level, maintenance_action

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ─── /api/status ──────────────────────────────────────────────────────────────

@api_bp.route("/status")
def status():
    """Return current model status and readiness flag."""
    return jsonify({"status": manager.status, "ready": manager.is_ready})


# ─── /api/train ───────────────────────────────────────────────────────────────

@api_bp.route("/train", methods=["POST"])
def train():
    """Trigger background model retraining."""
    if manager.status == "training":
        return jsonify({"message": "Training already in progress."}), 409
    manager.retrain_async()
    return jsonify({"message": "Training started in the background."})


# ─── /api/predict ─────────────────────────────────────────────────────────────

@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Run the full prediction pipeline for a single machine input.

    Expected JSON body:
        {
          "type":      "M",          // L, M, or H
          "air_temp":  300.5,        // Kelvin
          "proc_temp": 310.5,        // Kelvin
          "rpm":       1538,
          "torque":    40.0,         // Nm
          "tool_wear": 180           // minutes
        }
    """
    if not manager.is_ready:
        return jsonify({"error": "Models not ready. Please wait for training to finish."}), 503

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        validate_input(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = _run_pipeline(data)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── /api/simulate ────────────────────────────────────────────────────────────

@api_bp.route("/simulate")
def simulate():
    """
    Generate 8 random machine readings and return their predictions.
    Used by the dashboard's live monitoring panel.
    """
    if not manager.is_ready:
        return jsonify({"error": "Models not ready."}), 503

    rng     = np.random.default_rng()
    types   = ["L", "M", "H"]
    samples = []

    for i in range(8):
        air  = round(float(rng.uniform(297.0, 304.0)), 1)
        proc = round(float(rng.uniform(air + 7.5, min(air + 13.0, 313.8))), 1)

        rpm  = int(rng.choice([
            int(rng.integers(1168, 1400)),
            int(rng.integers(1400, 1700)),
            int(rng.integers(1700, 2886)),
        ]))
        torque    = round(float(rng.uniform(8.0, 74.0)), 1)
        tool_wear = int(rng.choice([
            int(rng.integers(0,   100)),
            int(rng.integers(100, 200)),
            int(rng.integers(180, 253)),
        ]))
        machine_type = str(rng.choice(types))

        inp    = {"type": machine_type, "air_temp": air, "proc_temp": proc,
                  "rpm": rpm, "torque": torque, "tool_wear": tool_wear}
        result = _run_pipeline(inp)
        result["machine_id"] = f"UNIT-{1000 + i}"
        result["input"]      = inp
        samples.append(result)

    return jsonify({"samples": samples})


# ─── Internal pipeline helper ─────────────────────────────────────────────────

def _run_pipeline(data: dict) -> dict:
    """
    Run the full prediction pipeline and return a serialisable result dict.

    Steps:
        1. Feature engineering
        2. Scale with the fitted StandardScaler
        3. MLP failure probability
        4. Autoencoder reconstruction error + anomaly flag
        5. Fusion risk score
        6. Leave-one-out feature attribution
        7. Maintenance action recommendation
    """
    pred_model = manager.predictor
    ae_model   = manager.anomaly_detector

    # 1 + 2: features → scaled
    raw_features  = build_features(data)
    sample_scaled = pred_model.transform(raw_features)

    # 3: classifier
    prob = float(pred_model.predict_proba_positive(sample_scaled)[0])
    pred = int(prob > pred_model.threshold)

    # 4: anomaly detection
    mse_val    = float(ae_model.score(sample_scaled)[0])
    is_anomaly = bool(mse_val > ae_model.threshold)

    # 5: fused risk score
    final_score = compute_risk_score(prob, mse_val, ae_model.threshold)

    # 6: feature attribution (top-2 for action, all for chart)
    all_importances  = get_feature_importance(sample_scaled, pred_model)
    top_causes       = all_importances[:2]
    top_feature_name = top_causes[0][0] if top_causes else ""

    # 7: maintenance action
    action = maintenance_action(pred, is_anomaly, top_feature_name)

    return {
        "prediction":        "FAILURE" if pred else "NORMAL",
        "probability":       round(prob * 100, 2),
        "risk_level":        risk_level(final_score),
        "confidence":        confidence_level(prob),
        "anomaly":           "YES" if is_anomaly else "NO",
        "anomaly_score":     round(mse_val, 6),
        "anomaly_threshold": round(ae_model.threshold, 6),
        "final_score":       round(final_score * 100, 2),
        "root_causes":       [{"feature": n, "impact": round(v, 6)} for n, v in top_causes],
        "action":            action,
        "feature_chart":     format_for_chart(all_importances),
    }
