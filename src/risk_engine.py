"""
risk_engine.py
--------------
Risk scoring, confidence classification, and maintenance action recommendation.

All logic mirrors the Honeywell notebook exactly:
- Risk score: 70% neural-network probability + 30% normalised reconstruction error
- Thresholds: HIGH > 0.7, MEDIUM > 0.4, LOW ≤ 0.4
- Actions: rule-based mapping from top root-cause feature + anomaly flag
"""


def compute_risk_score(prob: float, anomaly_mse: float, anomaly_threshold: float) -> float:
    """
    Fuse classifier probability and autoencoder anomaly score into a
    single risk score in the range [0, 1].

    Formula (from notebook):
        normalised_mse = min(anomaly_mse / anomaly_threshold, 1.0)
        risk_score     = 0.7 * prob + 0.3 * normalised_mse

    Parameters
    ----------
    prob              : Failure probability from the MLP (0–1).
    anomaly_mse       : Reconstruction error for the current sample.
    anomaly_threshold : 99th-percentile MSE threshold computed on normal data.

    Returns
    -------
    float in [0, 1]
    """
    normalised_mse = min(anomaly_mse / max(anomaly_threshold, 1e-9), 1.0)
    return 0.7 * prob + 0.3 * normalised_mse


def risk_level(score: float) -> str:
    """
    Map a risk score to a human-readable risk level.

    HIGH   : score > 0.70
    MEDIUM : score > 0.40
    LOW    : score ≤ 0.40
    """
    if score > 0.70:
        return "HIGH"
    if score > 0.40:
        return "MEDIUM"
    return "LOW"


def confidence_level(prob: float) -> str:
    """
    Classify prediction confidence based on distance from the decision boundary.

    UNCERTAIN        : |prob - 0.5| < 0.10  (close to the boundary)
    MODERATE         : otherwise
    HIGH CONFIDENCE  : prob > 0.80 or prob < 0.20
    """
    if abs(prob - 0.5) < 0.10:
        return "UNCERTAIN"
    if prob > 0.80 or prob < 0.20:
        return "HIGH CONFIDENCE"
    return "MODERATE"


def maintenance_action(pred: int, is_anomaly: bool, top_feature: str) -> str:
    """
    Produce a plain-language maintenance recommendation.

    Decision logic (from notebook):
    - If failure predicted, route to the most relevant action based on
      the top root-cause feature.
    - If no failure but anomaly detected, still flag for inspection.
    - Otherwise: normal operation.

    Parameters
    ----------
    pred        : Binary prediction (1 = FAILURE, 0 = NORMAL).
    is_anomaly  : Whether the autoencoder flagged this sample.
    top_feature : Name of the highest-impact feature from root-cause analysis.

    Returns
    -------
    str — maintenance action string
    """
    if pred == 1:
        if "Tool wear" in top_feature or "Wear_rate" in top_feature:
            return "Replace tool immediately — high wear detected"
        if "Torque" in top_feature or "Torque_per_speed" in top_feature:
            return "Check motor load — torque anomaly detected"
        if "temperature" in top_feature.lower() or "Temp_diff" in top_feature:
            return "Inspect cooling system — thermal deviation detected"
        if "Power" in top_feature:
            return "Check power supply and transmission — power anomaly"
        if is_anomaly:
            return "Unexpected anomaly — schedule full inspection"
        return "Schedule preventive maintenance"

    if is_anomaly:
        return "No failure predicted, but anomaly detected — monitor closely"

    return "Normal operation — continue monitoring"
