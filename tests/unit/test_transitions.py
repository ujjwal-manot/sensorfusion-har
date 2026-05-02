import numpy as np
import pytest
import torch
import torch.utils.data

from model.transitions import (
    detect_transitions,
    _classify_windows,
    evaluate_transition_accuracy,
)


class TestDetectTransitions:

    def test_simple(self):
        labels = [0, 0, 1, 1, 2]
        transitions = detect_transitions(labels)
        assert len(transitions) == 2
        assert transitions[0] == (2, 0, 1)
        assert transitions[1] == (4, 1, 2)

    def test_no_change(self):
        labels = [1, 1, 1]
        transitions = detect_transitions(labels)
        assert len(transitions) == 0

    def test_single_element(self):
        transitions = detect_transitions([5])
        assert len(transitions) == 0

    def test_every_step_changes(self):
        labels = [0, 1, 2, 3]
        transitions = detect_transitions(labels)
        assert len(transitions) == 3

    def test_empty(self):
        transitions = detect_transitions([])
        assert len(transitions) == 0


class TestClassifyWindows:

    def test_all_stable(self, synthetic_dataset):
        stable, trans, info = _classify_windows(synthetic_dataset)
        assert len(stable) + len(trans) == len(synthetic_dataset)

    def test_consistent_labels_all_stable(self):
        X = torch.randn(10, 128, 6)
        y = torch.ones(10, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, y)
        stable, trans, info = _classify_windows(ds)
        assert len(stable) == 10
        assert len(trans) == 0

    def test_alternating_labels_all_transition(self):
        X = torch.randn(6, 128, 6)
        y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, y)
        stable, trans, info = _classify_windows(ds)
        assert len(trans) == 6
        assert len(stable) == 0

    def test_raw_window_labels_branch(self):
        """Test the _raw_window_labels branch of _classify_windows."""
        X = torch.randn(5, 128, 6)
        y = torch.zeros(5, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, y)
        # Add _raw_window_labels attribute: all same label → all stable
        ds._raw_window_labels = [np.array([0] * 128) for _ in range(5)]
        stable, trans, info = _classify_windows(ds)
        assert len(stable) == 5
        assert len(trans) == 0

    def test_raw_window_labels_transition(self):
        """Test _raw_window_labels with mixed labels triggering transition."""
        X = torch.randn(3, 128, 6)
        y = torch.zeros(3, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, y)
        # Window 1: 50/50 split → below threshold
        mixed = np.array([0] * 64 + [1] * 64)
        ds._raw_window_labels = [
            np.array([0] * 128),  # stable
            mixed,                 # transition
            np.array([0] * 128),  # stable
        ]
        stable, trans, info = _classify_windows(ds, threshold=0.95)
        assert 1 in trans
        assert 0 in stable
        assert 2 in stable
        assert len(info) == 1


class TestEvaluateTransitionAccuracy:

    @pytest.fixture
    def simple_dataset(self):
        X = torch.randn(20, 128, 6)
        y = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
        return torch.utils.data.TensorDataset(X, y)

    def test_returns_expected_keys(self, model, simple_dataset, device):
        results = evaluate_transition_accuracy(model, simple_dataset, device)
        assert "stable_accuracy" in results
        assert "transition_accuracy" in results
        assert "stable_count" in results
        assert "transition_count" in results
        assert "pair_accuracy" in results
        assert "overall_accuracy" in results
        assert "num_classes" in results

    def test_counts_sum_to_total(self, model, simple_dataset, device):
        results = evaluate_transition_accuracy(model, simple_dataset, device)
        assert results["stable_count"] + results["transition_count"] == len(simple_dataset)

    def test_all_stable_dataset(self, model, device):
        X = torch.randn(10, 128, 6)
        y = torch.zeros(10, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, y)
        results = evaluate_transition_accuracy(model, ds, device)
        assert results["stable_count"] == 10
        assert results["transition_count"] == 0
        assert results["transition_accuracy"] == 0.0

    def test_accuracy_range(self, model, simple_dataset, device):
        results = evaluate_transition_accuracy(model, simple_dataset, device)
        assert 0.0 <= results["overall_accuracy"] <= 1.0
        assert 0.0 <= results["stable_accuracy"] <= 1.0
