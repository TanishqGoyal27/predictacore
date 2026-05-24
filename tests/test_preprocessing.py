"""Tests for feature engineering and dataset generation."""

import numpy as np
import pytest

from src.preprocessing import (
    build_features,
    generate_dataset,
    validate_input,
    FEATURE_NAMES,
)

VALID_INPUT = {
    "type": "M",
    "air_temp": 300.5,
    "proc_temp": 310.5,
    "rpm": 1538,
    "torque": 40.0,
    "tool_wear": 180,
}


class TestValidateInput:
    def test_valid_input_passes(self):
        validate_input(VALID_INPUT)  # should not raise

    def test_missing_field_raises(self):
        bad = {k: v for k, v in VALID_INPUT.items() if k != "rpm"}
        with pytest.raises(ValueError, match="rpm"):
            validate_input(bad)

    def test_invalid_type_raises(self):
        bad = {**VALID_INPUT, "type": "X"}
        with pytest.raises(ValueError, match="type"):
            validate_input(bad)

    def test_out_of_range_rpm_raises(self):
        bad = {**VALID_INPUT, "rpm": 9999}
        with pytest.raises(ValueError, match="rpm"):
            validate_input(bad)


class TestBuildFeatures:
    def test_output_shape(self):
        X = build_features(VALID_INPUT)
        assert X.shape == (1, len(FEATURE_NAMES))

    def test_dtype_float64(self):
        X = build_features(VALID_INPUT)
        assert X.dtype == np.float64

    def test_derived_features_correct(self):
        X = build_features(VALID_INPUT)
        air, proc, rpm, torque, wear = 300.5, 310.5, 1538.0, 40.0, 180.0

        temp_diff        = proc - air
        power            = rpm * torque
        torque_per_speed = torque / (rpm + 1.0)
        wear_rate        = wear / (rpm + 1.0)

        assert X[0, 6] == pytest.approx(temp_diff)
        assert X[0, 7] == pytest.approx(power)
        assert X[0, 8] == pytest.approx(torque_per_speed)
        assert X[0, 9] == pytest.approx(wear_rate)

    def test_type_encoding(self):
        for raw, expected in [("L", 0), ("M", 1), ("H", 2)]:
            X = build_features({**VALID_INPUT, "type": raw})
            assert X[0, 0] == expected


class TestGenerateDataset:
    def test_output_shapes(self):
        X, y = generate_dataset(n=200, seed=0)
        assert X.shape == (200, len(FEATURE_NAMES))
        assert y.shape == (200,)

    def test_binary_labels(self):
        _, y = generate_dataset(n=500, seed=0)
        assert set(np.unique(y)).issubset({0, 1})

    def test_failure_rate_plausible(self):
        _, y = generate_dataset(n=5000, seed=42)
        rate = y.mean()
        # ai4i2020 has ~3.4% failures; our synthetic dataset is higher
        # due to multiple overlapping failure rules — check it's non-trivial
        assert 0.01 < rate < 0.5, f"Unexpected failure rate: {rate:.3f}"
