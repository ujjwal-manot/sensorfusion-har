import torch
import pytest
from model.attention import PatchMicroAttention


class TestPatchMicroAttention:

    def test_output_shape(self, attention):
        x = torch.randn(4, 48, 32)
        out = attention(x)
        assert out.shape == (4, 32)

    def test_return_attention(self, attention):
        x = torch.randn(4, 48, 32)
        pooled, weights, entropy = attention(x, return_attention=True)
        assert pooled.shape == (4, 32)
        assert weights.shape == (4, 2, 8, 8)  # (batch, heads, patches, patches)
        assert entropy.shape == ()  # scalar

    def test_entropy_nonnegative(self, attention):
        x = torch.randn(4, 48, 32)
        _, _, entropy = attention(x, return_attention=True)
        assert entropy.item() >= 0.0

    def test_positional_embedding_shape(self, attention):
        assert attention.pos_embedding.shape == (1, 8, 32)

    def test_patch_dimension(self, attention):
        # in_channels=48, seq_len=32, num_patches=8 -> patch_size=4, patch_dim=192
        assert attention.patch_dim == 48 * 4

    def test_gradients_flow(self, attention):
        x = torch.randn(4, 48, 32, requires_grad=True)
        out = attention(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert attention.projection.weight.grad is not None

    def test_attention_weights_sum_to_one(self, attention):
        x = torch.randn(4, 48, 32)
        _, weights, _ = attention(x, return_attention=True)
        # Each head's attention weights should sum to ~1 along last dim
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
