"""Tests for visualization module. Uses Agg backend (no display)."""
import numpy as np
import pytest
import torch
import torch.utils.data
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from model.visualize import (
    _get_labels_and_colors,
    _extract_embeddings,
    plot_tsne,
    plot_attention_maps,
    plot_noise_robustness,
    plot_confidence_calibration,
)


class TestGetLabelsAndColors:

    def test_default_6_classes(self):
        labels, colors = _get_labels_and_colors(None, None, 6)
        assert len(labels) == 6
        assert len(colors) == 6

    def test_custom_labels(self):
        labels, colors = _get_labels_and_colors(["A", "B"], None, 2)
        assert labels == ["A", "B"]

    def test_many_classes_fallback(self):
        labels, colors = _get_labels_and_colors(None, None, 12)
        assert len(labels) == 12
        assert len(colors) == 12
        assert labels[0] == "Class 0"

    def test_custom_colors(self):
        labels, colors = _get_labels_and_colors(None, ["red", "blue"], 2)
        assert colors == ["red", "blue"]


class TestExtractEmbeddings:

    @pytest.fixture
    def small_dataset(self):
        X = torch.randn(20, 128, 6)
        y = torch.randint(0, 3, (20,))
        return torch.utils.data.TensorDataset(X, y)

    def test_returns_stages_and_labels(self, model, small_dataset, device):
        stages, labels = _extract_embeddings(model, small_dataset, device, n_samples=10)
        assert "reservoir" in stages
        assert "dsconv" in stages
        assert "attention" in stages
        assert "final" in stages
        assert len(labels) == 10

    def test_n_samples_cap(self, model, small_dataset, device):
        stages, labels = _extract_embeddings(model, small_dataset, device, n_samples=1000)
        assert len(labels) == 20  # capped at dataset size


class TestPlotTsne:

    @pytest.fixture
    def small_dataset(self):
        X = torch.randn(20, 128, 6)
        y = torch.randint(0, 3, (20,))
        return torch.utils.data.TensorDataset(X, y)

    def test_all_stages(self, model, small_dataset, device):
        fig = plot_tsne(model, small_dataset, device, stage="all", n_samples=20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_stage(self, model, small_dataset, device):
        fig = plot_tsne(model, small_dataset, device, stage="attention", n_samples=20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotAttentionMaps:

    @pytest.fixture
    def small_dataset(self):
        X = torch.randn(10, 128, 6)
        y = torch.randint(0, 3, (10,))
        return torch.utils.data.TensorDataset(X, y)

    def test_returns_figure(self, model, small_dataset, device):
        fig = plot_attention_maps(model, small_dataset, device, n_samples=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_sample(self, model, small_dataset, device):
        fig = plot_attention_maps(model, small_dataset, device, n_samples=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotNoiseRobustness:

    @pytest.fixture
    def small_dataset(self):
        X = torch.randn(16, 128, 6)
        y = torch.randint(0, 3, (16,))
        return torch.utils.data.TensorDataset(X, y)

    def test_returns_results_and_figure(self, model, small_dataset, device):
        results, fig = plot_noise_robustness(
            model, small_dataset, device, snr_levels=[40, 20]
        )
        assert isinstance(fig, plt.Figure)
        assert "accuracy" in results
        assert "f1" in results
        assert len(results["accuracy"]) == 2
        plt.close(fig)

    def test_custom_labels(self, model, small_dataset, device):
        results, fig = plot_noise_robustness(
            model, small_dataset, device, snr_levels=[40],
            activity_labels=["A", "B", "C"]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotConfidenceCalibration:

    @pytest.fixture
    def small_dataset(self):
        X = torch.randn(16, 128, 6)
        y = torch.randint(0, 6, (16,))
        return torch.utils.data.TensorDataset(X, y)

    def test_returns_ece_and_figure(self, model, small_dataset, device):
        ece, fig = plot_confidence_calibration(model, small_dataset, device, n_bins=5)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_default_bins(self, model, small_dataset, device):
        ece, fig = plot_confidence_calibration(model, small_dataset, device)
        assert isinstance(ece, float)
        plt.close(fig)
