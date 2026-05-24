"""
preprocessing.py
----------------
Feature engineering, input validation, and synthetic dataset generation.

All transformations mirror the original Honeywell notebook exactly:
- Type encoding: L→0, M→1, H→2
- Derived features: Temp_diff, Power, Torque_per_speed, Wear_rate
- Synthetic dataset follows the ai4i2020 statistical distribution
"""

import numpy as np
import pandas as pd
from typing import Dict


# Ordered list of features fed into the ML models.
FEATURE_NAMES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_diff",
    "Power",
    "Torque_per_speed",
    "Wear_rate",
]

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Valid sensor ranges from the ai4i2020 dataset
SENSOR_RANGES = {
    "air_temp":  (295.3, 304.5),
    "proc_temp": (305.7, 313.8),
    "rpm":       (1168,  2886),
    "torque":    (3.8,   76.6),
    "tool_wear": (0,     253),
}


def validate_input(data: Dict) -> None:
    """
    Raise ValueError if any required field is missing or out of range.
    Call this before build_features() to give helpful API error messages.
    """
    required = ["type", "air_temp", "proc_temp", "rpm", "torque", "tool_wear"]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: '{field}'")

    if str(data["type"]).upper() not in TYPE_MAP:
        raise ValueError(f"'type' must be one of {list(TYPE_MAP.keys())}")

    for key, (lo, hi) in SENSOR_RANGES.items():
        val = float(data[key])
        if not (lo <= val <= hi):
            raise ValueError(f"'{key}' value {val} is outside valid range [{lo}, {hi}]")


def build_features(row: Dict) -> np.ndarray:
    """
    Convert a raw API input dict into a (1, 10) feature array.

    Derived features (identical to notebook):
        Temp_diff        = proc_temp - air_temp
        Power            = rpm * torque
        Torque_per_speed = torque / (rpm + 1)
        Wear_rate        = tool_wear / (rpm + 1)

    Returns
    -------
    np.ndarray of shape (1, 10), dtype float64
    """
    t         = float(TYPE_MAP.get(str(row["type"]).upper(), 1))
    air_temp  = float(row["air_temp"])
    proc_temp = float(row["proc_temp"])
    rpm       = float(row["rpm"])
    torque    = float(row["torque"])
    tool_wear = float(row["tool_wear"])

    temp_diff        = proc_temp - air_temp
    power            = rpm * torque
    torque_per_speed = torque / (rpm + 1.0)
    wear_rate        = tool_wear / (rpm + 1.0)

    return np.array(
        [[t, air_temp, proc_temp, rpm, torque, tool_wear,
          temp_diff, power, torque_per_speed, wear_rate]],
        dtype=np.float64,
    )


def generate_dataset(n: int = 10_000, seed: int = 42):
    """
    Generate a synthetic dataset that mirrors the ai4i2020 statistical distribution
    and failure rules described in the dataset paper.

    Parameters
    ----------
    n    : Number of samples to generate.
    seed : Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n, 10) — feature matrix
    y : np.ndarray of shape (n,)   — binary failure labels
    """
    rng = np.random.default_rng(seed)

    air_temp  = rng.normal(300.0, 2.0,   n).clip(*SENSOR_RANGES["air_temp"])
    proc_temp = (air_temp + rng.normal(10.0, 1.5, n)).clip(*SENSOR_RANGES["proc_temp"])
    rpm       = rng.normal(1538.0, 179.0, n).clip(*SENSOR_RANGES["rpm"])
    torque    = rng.normal(40.0,   9.97,  n).clip(*SENSOR_RANGES["torque"])
    tool_wear = rng.integers(0, 254, n).astype(float)
    type_col  = rng.choice([0.0, 1.0, 2.0], n, p=[0.6, 0.3, 0.1])

    temp_diff        = proc_temp - air_temp
    power            = rpm * torque
    torque_per_speed = torque / (rpm + 1.0)
    wear_rate        = tool_wear / (rpm + 1.0)

    X = np.column_stack([
        type_col, air_temp, proc_temp, rpm, torque, tool_wear,
        temp_diff, power, torque_per_speed, wear_rate,
    ])

    # Failure labelling rules from the ai4i2020 dataset paper
    y = np.zeros(n, dtype=int)

    # Tool Wear Failure (TWF) — thresholds differ by machine type
    y[(tool_wear > 200) & (type_col == 0)] = 1
    y[(tool_wear > 150) & (type_col == 1)] = 1
    y[(tool_wear > 120) & (type_col == 2)] = 1

    # Heat Dissipation Failure (HDF)
    y[(temp_diff < 8.6) & (rpm < 1380)] = 1

    # Power Failure (PWF) — angular power outside [3500, 9000] W
    angular_power = torque * rpm * (2 * np.pi / 60)
    y[angular_power < 3500] = 1
    y[angular_power > 9000] = 1

    # Overstrain Failure (OSF) — torque × wear exceeds type-specific limits
    strain = torque * tool_wear
    y[(type_col == 0) & (strain > 11_000)] = 1
    y[(type_col == 1) & (strain > 8_000)]  = 1
    y[(type_col == 2) & (strain > 6_000)]  = 1

    # Random Noise Failures (RNF) — 0.2% random
    y[rng.random(n) < 0.002] = 1

    return X, y
