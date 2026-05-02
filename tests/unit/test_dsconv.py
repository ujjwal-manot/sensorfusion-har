import torch
import pytest
from model.dsconv import DSConvEncoder, DepthwiseSeparableBlock


class TestDepthwiseSeparableBlock:

    def test_output_shape(self):
        block = DepthwiseSeparableBlock(32, 48, kernel_size=5, stride=1, padding=2)
        x = torch.randn(4, 32, 128)
        out = block(x)
        assert out.shape == (4, 48, 128)

    def test_depthwise_groups(self):
        block = DepthwiseSeparableBlock(32, 48, kernel_size=5, stride=1, padding=2)
        assert block.depthwise.groups == 32

    def test_stride_reduces_length(self):
        block = DepthwiseSeparableBlock(32, 48, kernel_size=5, stride=2, padding=2)
        x = torch.randn(4, 32, 128)
        out = block(x)
        assert out.shape[2] == 64


class TestDSConvEncoder:

    def test_output_shape(self, dsconv_encoder):
        x = torch.randn(4, 32, 128)
        out = dsconv_encoder(x)
        # stride=1, stride=2, stride=2 -> 128 / 4 = 32
        assert out.shape == (4, 48, 32)

    def test_gradients_flow(self, dsconv_encoder):
        x = torch.randn(4, 32, 128, requires_grad=True)
        out = dsconv_encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_batchnorm_updates_in_train(self, dsconv_encoder):
        dsconv_encoder.train()
        x = torch.randn(4, 32, 128)
        first_block = dsconv_encoder.blocks[0]
        mean_before = first_block.bn.running_mean.clone()
        dsconv_encoder(x)
        mean_after = first_block.bn.running_mean
        assert not torch.allclose(mean_before, mean_after)
