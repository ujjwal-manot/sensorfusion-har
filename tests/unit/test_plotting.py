"""Tests for plotting functions. Uses Agg backend (no display)."""
import pytest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class TestTransitionPlot:

    def test_plot_returns_figure(self):
        from model.transitions import plot_transition_analysis
        results = {
            "stable_accuracy": 0.9,
            "transition_accuracy": 0.6,
            "overall_accuracy": 0.8,
            "stable_count": 100,
            "transition_count": 20,
            "pair_accuracy": {(0, 1): 0.7, (1, 2): 0.5},
            "num_classes": 6,
        }
        fig = plot_transition_analysis(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_no_pairs(self):
        from model.transitions import plot_transition_analysis
        results = {
            "stable_accuracy": 0.9,
            "transition_accuracy": 0.0,
            "overall_accuracy": 0.85,
            "stable_count": 100,
            "transition_count": 0,
            "pair_accuracy": {},
            "num_classes": 6,
        }
        fig = plot_transition_analysis(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_custom_labels(self):
        from model.transitions import plot_transition_analysis
        results = {
            "stable_accuracy": 0.9,
            "transition_accuracy": 0.6,
            "overall_accuracy": 0.8,
            "stable_count": 50,
            "transition_count": 10,
            "pair_accuracy": {(0, 1): 0.7},
            "num_classes": 3,
        }
        fig = plot_transition_analysis(results, activity_labels=["A", "B", "C"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_many_classes(self):
        from model.transitions import plot_transition_analysis
        results = {
            "stable_accuracy": 0.9,
            "transition_accuracy": 0.6,
            "overall_accuracy": 0.8,
            "stable_count": 50,
            "transition_count": 10,
            "pair_accuracy": {},
            "num_classes": 12,
        }
        fig = plot_transition_analysis(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestAdversarialPlot:

    def test_plot_both_attacks(self):
        from model.adversarial import plot_adversarial_robustness
        results = {
            "epsilons": [0, 0.05, 0.1],
            "clean_accuracy": 0.85,
            "fgsm_accuracy": [0.85, 0.6, 0.4],
            "pgd_accuracy": [0.85, 0.5, 0.3],
        }
        fig = plot_adversarial_robustness(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fgsm_only(self):
        from model.adversarial import plot_adversarial_robustness
        results = {
            "epsilons": [0, 0.1],
            "clean_accuracy": 0.8,
            "fgsm_accuracy": [0.8, 0.5],
            "pgd_accuracy": [],
        }
        fig = plot_adversarial_robustness(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestDriftPlot:

    def test_plot_bias_only(self):
        from model.drift import plot_drift_robustness
        results = {
            "clean_accuracy": 0.9,
            "drift_types": ["bias"],
            "drift_levels": {"bias": [0, 0.01, 0.05]},
            "bias_accuracy": [0.9, 0.85, 0.7],
        }
        fig = plot_drift_robustness(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_all_types(self):
        from model.drift import plot_drift_robustness
        results = {
            "clean_accuracy": 0.9,
            "drift_types": ["bias", "scale", "noise"],
            "drift_levels": {
                "bias": [0, 0.01],
                "scale": [0, 0.001],
                "noise": [(40, 40), (40, 10)],
            },
            "bias_accuracy": [0.9, 0.8],
            "scale_accuracy": [0.9, 0.85],
            "noise_accuracy": [0.9, 0.6],
        }
        fig = plot_drift_robustness(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestEnergyPlot:

    def test_plot_comparison(self):
        from model.energy import plot_energy_comparison
        import numpy as np
        comparison = {
            "Model A": {
                "total_macs": 1000000,
                "total_params": 23000,
                "energy_fp32_mj": 0.0046,
                "energy_int8_mj": 0.00023,
                "macs_per_layer": {"layer1": 500000, "layer2": 500000},
                "energy_per_layer_fp32": {"layer1": 0.0023, "layer2": 0.0023},
            },
            "Model B": {
                "total_macs": 2000000,
                "total_params": 50000,
                "energy_fp32_mj": 0.0092,
                "energy_int8_mj": 0.00046,
                "macs_per_layer": {"layer1": 1000000, "layer2": 1000000},
                "energy_per_layer_fp32": {"layer1": 0.0046, "layer2": 0.0046},
            },
        }
        fig = plot_energy_comparison(comparison)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
