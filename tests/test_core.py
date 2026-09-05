import numpy as np
import pytest

from src.features import FEATURE_COLUMNS
from src.generate_data import generate
from src.train import evaluate_at_threshold


def test_generated_data_is_reproducible_and_well_formed():
    first = generate(1000, 0.02)
    second = generate(1000, 0.02)
    assert first.equals(second)
    assert set(FEATURE_COLUMNS + ["is_fraud"]).issubset(first.columns)
    assert first["is_fraud"].sum() == 20


def test_threshold_metrics_handle_all_negative_predictions():
    metrics = evaluate_at_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]), 0.9)
    assert metrics["true_negatives"] == 4 - 2
    assert metrics["false_negatives"] == 2
    assert metrics["precision"] == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_threshold_is_bounded(threshold):
    with pytest.raises(ValueError):
        evaluate_at_threshold(np.array([0, 1]), np.array([0.1, 0.9]), threshold)


@pytest.mark.parametrize("args", [(9, 0.02), (100, 0), (100, 1)])
def test_generator_rejects_invalid_sizes(args):
    with pytest.raises(ValueError):
        generate(*args)
