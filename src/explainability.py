"""
explainability.py
-----------------
Feature attribution via leave-one-out perturbation.

This is NOT a certified XAI method (e.g. SHAP or LIME). It is a simple
sensitivity analysis that measures how much the model's failure probability
changes when each feature is zeroed out. It is computationally cheap and
gives useful directional insight for the dashboard, but should not be
treated as a rigorous attribution.

The logic is a direct translation of get_root_cause() from the notebook.
"""

from typing import List, Tuple

import numpy as np

from src.preprocessing import FEATURE_NAMES


def get_feature_importance(
    sample_scaled: np.ndarray,
    predictor,
    top_k: int = None,
) -> List[Tuple[str, float]]:
    """
    Estimate feature importance for a single sample via leave-one-out
    perturbation (zero-ablation).

    For each feature i, the feature value is set to 0.0 (in scaled space)
    and the change in failure probability is recorded. Larger change →
    higher importance.

    Parameters
    ----------
    sample_scaled : np.ndarray of shape (1, n_features) — already scaled.
    predictor     : FailurePredictor instance with predict_proba_positive().
    top_k         : Return only the top-k features. None returns all.

    Returns
    -------
    List of (feature_name, impact) tuples, sorted descending by impact.
    """
    baseline   = float(predictor.predict_proba_positive(sample_scaled)[0])
    importances: List[Tuple[str, float]] = []

    for i, name in enumerate(FEATURE_NAMES):
        modified       = sample_scaled.copy()
        modified[0, i] = 0.0
        new_prob = float(predictor.predict_proba_positive(modified)[0])
        impact   = round(abs(baseline - new_prob), 6)
        importances.append((name, impact))

    importances.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        return importances[:top_k]
    return importances


def format_for_chart(importances: List[Tuple[str, float]], top_n: int = 6) -> List[dict]:
    """
    Convert the raw importance list into a JSON-serialisable format
    for the frontend bar chart.

    Normalises impact values to a 0–100 scale relative to the top feature.

    Parameters
    ----------
    importances : Output from get_feature_importance().
    top_n       : Number of features to include in the chart.

    Returns
    -------
    List of {"feature": str, "impact": float} dicts.
    """
    top = importances[:top_n]
    if not top:
        return []

    max_impact = max(v for _, v in top) or 1.0
    return [
        {"feature": name, "impact": round((v / max_impact) * 100, 1)}
        for name, v in top
    ]
