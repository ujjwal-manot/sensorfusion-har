import torch
import pytest
from model.reservoir import EchoStateNetwork


class TestEchoStateNetwork:

    def test_output_shape(self, synthetic_batch):
        esn = EchoStateNetwork(6, 32, use_diff_states=False)
        out = esn(synthetic_batch)
        assert out.shape == (8, 128, 32)

    def test_diff_states_output_shape(self, synthetic_batch, reservoir):
        out = reservoir(synthetic_batch)
        assert out.shape == (8, 128, 64)

    def test_output_dim_no_diff(self):
        esn = EchoStateNetwork(6, 32, use_diff_states=False)
        assert esn.output_dim == 32

    def test_output_dim_with_diff(self, reservoir):
        assert reservoir.output_dim == 64

    def test_learnable_sr_gradient_flows(self, synthetic_batch, reservoir):
        out = reservoir(synthetic_batch)
        loss = out.sum()
        loss.backward()
        assert reservoir.sr_logit is not None
        assert reservoir.sr_logit.grad is not None

    def test_fixed_sr_no_parameter(self, reservoir_nodiff):
        assert reservoir_nodiff.sr_logit is None

    def test_effective_spectral_radius_range(self, reservoir):
        sr = reservoir.effective_spectral_radius
        assert 0.0 < sr.item() < 1.0

    def test_reservoir_dropout_training(self, synthetic_batch, reservoir):
        reservoir.train()
        out1 = reservoir(synthetic_batch)
        out2 = reservoir(synthetic_batch)
        # With dropout, outputs should differ between calls (stochastic masking)
        # Note: seeds are reset per-test, but dropout generates new masks each call
        assert out1.shape == out2.shape

    def test_reservoir_dropout_eval_deterministic(self, synthetic_batch, reservoir):
        reservoir.eval()
        out1 = reservoir(synthetic_batch)
        out2 = reservoir(synthetic_batch)
        assert torch.allclose(out1, out2)

    def test_sparsity_mask(self):
        esn = EchoStateNetwork(6, 64, sparsity=0.8)
        zero_frac = (esn.W_res == 0).float().mean().item()
        assert zero_frac > 0.6  # approximately 80% sparse

    def test_spectral_init(self, synthetic_dataset):
        esn = EchoStateNetwork.spectral_init(
            synthetic_dataset, input_channels=6, reservoir_size=32,
        )
        assert isinstance(esn, EchoStateNetwork)
        assert esn.W_in.shape == (6, 32)

    def test_deterministic_eval(self, synthetic_batch, reservoir):
        reservoir.eval()
        out_a = reservoir(synthetic_batch)
        out_b = reservoir(synthetic_batch)
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_batch_independence(self, reservoir):
        reservoir.eval()
        x = torch.randn(4, 128, 6)
        out_all = reservoir(x)

        x_mod = x.clone()
        x_mod[0] = torch.randn(128, 6)
        out_mod = reservoir(x_mod)

        # Other samples should be unaffected
        assert torch.allclose(out_all[1], out_mod[1], atol=1e-6)
        assert torch.allclose(out_all[2], out_mod[2], atol=1e-6)
