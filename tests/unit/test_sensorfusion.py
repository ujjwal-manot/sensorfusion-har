import torch
import pytest
from model.sensorfusion import SensorFusionHAR, GatedResidualFusion, SpectralGatedFusion


class TestSensorFusionHAR:

    def test_forward_output_shape(self, model, synthetic_batch):
        out = model(synthetic_batch)
        assert out.shape == (8, 6)

    def test_forward_with_aux(self, model, synthetic_batch):
        logits, aux = model(synthetic_batch, return_aux=True)
        assert logits.shape == (8, 6)
        assert "attention_entropy" in aux
        assert "attention_weights" in aux
        assert "spectral_radius" in aux

    def test_forward_from_reservoir(self, model, synthetic_batch):
        h = model.reservoir_states(synthetic_batch)
        out = model.forward_from_reservoir(h)
        assert out.shape == (8, 6)

    def test_reservoir_states(self, model, synthetic_batch):
        h = model.reservoir_states(synthetic_batch)
        assert h.shape == (8, 128, 64)  # diff_states doubles dim

    def test_merge_reservoir_states_with_diff(self, model, synthetic_batch):
        h = model.reservoir_states(synthetic_batch)
        merged = model._merge_reservoir_states(h)
        assert merged.shape == (8, 128, 32)

    def test_merge_reservoir_states_no_diff(self):
        m = SensorFusionHAR.__new__(SensorFusionHAR)
        torch.nn.Module.__init__(m)
        m.diff_gate = None
        h = torch.randn(4, 128, 32)
        assert torch.equal(m._merge_reservoir_states(h), h)

    def test_count_parameters(self, model):
        count = model.count_parameters()
        assert isinstance(count, int)
        assert count > 0

    def test_model_size_kb(self, model):
        size = model.model_size_kb()
        assert size == model.count_parameters() * 4 / 1024

    def test_quantized_size_kb(self, model):
        size = model.quantized_size_kb()
        assert size == model.count_parameters() * 1 / 1024

    def test_architecture_summary_keys(self, model):
        summary = model.architecture_summary()
        expected = {
            "learnable_spectral_radius", "effective_sr", "differential_states",
            "reservoir_dropout", "fusion_type", "scaled_binary_weights",
            "trainable_parameters", "model_size_fp32_kb", "model_size_int8_kb",
        }
        assert set(summary.keys()) == expected

    def test_architecture_summary_values(self, model):
        summary = model.architecture_summary()
        assert summary["learnable_spectral_radius"] is True
        assert summary["differential_states"] is True
        assert summary["fusion_type"] == "SpectralGatedFusion"
        assert summary["scaled_binary_weights"] is True

    def test_quantize_returns_module(self, model):
        quantized = model.quantize()
        assert isinstance(quantized, torch.nn.Module)

    def test_quantize_forward_works(self, model, synthetic_batch):
        quantized = model.quantize()
        out = quantized(synthetic_batch)
        assert out.shape == (8, 6)

    def test_12_class_config(self, model_12cls, synthetic_batch):
        out = model_12cls(synthetic_batch)
        assert out.shape == (8, 12)

    def test_gradient_flow_end_to_end(self, model, synthetic_batch, synthetic_labels):
        out = model(synthetic_batch)
        loss = torch.nn.functional.cross_entropy(out, synthetic_labels)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestGatedResidualFusion:

    def test_output_shape(self):
        fusion = GatedResidualFusion(reservoir_dim=32, dsconv_channels=48, seq_len=32)
        reservoir_out = torch.randn(4, 32, 128)
        dsconv_out = torch.randn(4, 48, 32)
        out = fusion(reservoir_out, dsconv_out)
        assert out.shape == (4, 48, 32)


class TestSpectralGatedFusion:

    def test_output_shape(self):
        fusion = SpectralGatedFusion(reservoir_dim=32, dsconv_channels=48, seq_len=32)
        reservoir_out = torch.randn(4, 32, 128)
        dsconv_out = torch.randn(4, 48, 32)
        out = fusion(reservoir_out, dsconv_out)
        assert out.shape == (4, 48, 32)

    def test_differs_from_simple_addition(self):
        fusion = SpectralGatedFusion(reservoir_dim=32, dsconv_channels=48, seq_len=32)
        reservoir_out = torch.randn(4, 32, 128)
        dsconv_out = torch.randn(4, 48, 32)
        out = fusion(reservoir_out, dsconv_out)
        # Should not be identical to just dsconv_out
        assert not torch.allclose(out, dsconv_out, atol=1e-3)
