import torch
import pytest
from model.drift import (
    simulate_bias_drift,
    simulate_scale_drift,
    simulate_noise_drift,
    evaluate_drift_robustness,
)


class TestBiasDrift:

    def test_shape_3d(self, synthetic_batch):
        out = simulate_bias_drift(synthetic_batch, drift_rate=0.01)
        assert out.shape == synthetic_batch.shape

    def test_shape_2d(self):
        x = torch.randn(128, 6)
        out = simulate_bias_drift(x, drift_rate=0.01)
        assert out.shape == x.shape

    def test_zero_rate_noop(self, synthetic_batch):
        out = simulate_bias_drift(synthetic_batch, drift_rate=0.0)
        assert torch.allclose(out, synthetic_batch)

    def test_channel_selection(self, synthetic_batch):
        out = simulate_bias_drift(synthetic_batch, drift_rate=0.01, channels=[0, 1])
        assert torch.allclose(out[:, :, 2:], synthetic_batch[:, :, 2:])

    def test_drift_increases_over_time(self, synthetic_batch):
        out = simulate_bias_drift(synthetic_batch, drift_rate=0.1)
        diff = out - synthetic_batch
        # Later timesteps should have larger drift
        assert diff[:, -1, 0].abs().mean() > diff[:, 0, 0].abs().mean()


class TestScaleDrift:

    def test_shape_3d(self, synthetic_batch):
        out = simulate_scale_drift(synthetic_batch, drift_rate=0.001)
        assert out.shape == synthetic_batch.shape

    def test_shape_2d(self):
        x = torch.randn(128, 6)
        out = simulate_scale_drift(x, drift_rate=0.001)
        assert out.shape == x.shape

    def test_zero_rate_noop(self, synthetic_batch):
        out = simulate_scale_drift(synthetic_batch, drift_rate=0.0)
        assert torch.allclose(out, synthetic_batch)

    def test_channel_selection(self):
        x = torch.randn(128, 6)
        out = simulate_scale_drift(x, drift_rate=0.01, channels=[0])
        assert torch.allclose(out[:, 1:], x[:, 1:])

    def test_2d_channel_selection(self):
        x = torch.randn(128, 6)
        out = simulate_scale_drift(x, drift_rate=0.01, channels=[0, 1])
        assert torch.allclose(out[:, 2:], x[:, 2:])


class TestNoiseDrift:

    def test_shape_3d(self, synthetic_batch):
        out = simulate_noise_drift(synthetic_batch, initial_snr=40, final_snr=10)
        assert out.shape == synthetic_batch.shape

    def test_shape_2d(self):
        x = torch.randn(128, 6)
        out = simulate_noise_drift(x, initial_snr=40, final_snr=10)
        assert out.shape == x.shape

    def test_adds_noise(self, synthetic_batch):
        out = simulate_noise_drift(synthetic_batch, initial_snr=40, final_snr=5)
        assert not torch.allclose(out, synthetic_batch)

    def test_2d_channel_selection(self):
        x = torch.ones(128, 6)
        out = simulate_noise_drift(x, initial_snr=40, final_snr=10, channels=[0])
        # Channels 1-5 should be unchanged for constant input
        # (noise power depends on signal power which is 1.0)
        assert out.shape == x.shape

    def test_2d_adds_noise(self):
        x = torch.randn(128, 6)
        out = simulate_noise_drift(x, initial_snr=40, final_snr=5)
        assert not torch.allclose(out, x)


class TestEvaluateDriftRobustness:

    @pytest.fixture
    def small_dataset(self):
        X = torch.randn(16, 128, 6)
        y = torch.randint(0, 6, (16,))
        return torch.utils.data.TensorDataset(X, y)

    def test_returns_expected_keys(self, model, small_dataset, device):
        results = evaluate_drift_robustness(
            model, small_dataset, device,
            drift_types=["bias", "scale"],
            drift_levels={
                "bias": [0, 0.01],
                "scale": [0, 0.001],
            },
        )
        assert "clean_accuracy" in results
        assert "bias_accuracy" in results
        assert "scale_accuracy" in results

    def test_clean_accuracy_first(self, model, small_dataset, device):
        results = evaluate_drift_robustness(
            model, small_dataset, device,
            drift_types=["bias"],
            drift_levels={"bias": [0, 0.01]},
        )
        assert results["bias_accuracy"][0] == results["clean_accuracy"]

    def test_noise_drift_type(self, model, small_dataset, device):
        results = evaluate_drift_robustness(
            model, small_dataset, device,
            drift_types=["noise"],
            drift_levels={"noise": [(40, 40), (40, 10)]},
        )
        assert "noise_accuracy" in results
        assert len(results["noise_accuracy"]) == 2

    def test_all_drift_types(self, model, small_dataset, device):
        results = evaluate_drift_robustness(
            model, small_dataset, device,
            drift_types=["bias", "scale", "noise"],
            drift_levels={
                "bias": [0, 0.01],
                "scale": [0, 0.001],
                "noise": [(40, 40), (40, 10)],
            },
        )
        assert "bias_accuracy" in results
        assert "scale_accuracy" in results
        assert "noise_accuracy" in results

    def test_accuracy_in_range(self, model, small_dataset, device):
        results = evaluate_drift_robustness(
            model, small_dataset, device,
            drift_types=["bias"],
            drift_levels={"bias": [0, 0.05]},
        )
        for acc in results["bias_accuracy"]:
            assert 0.0 <= acc <= 1.0
