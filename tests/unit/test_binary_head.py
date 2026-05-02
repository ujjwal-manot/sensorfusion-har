import torch
import pytest
from model.binary_head import BinaryLinear, BinaryClassifier


class TestBinaryLinear:

    def test_output_shape(self):
        bl = BinaryLinear(32, 6)
        x = torch.randn(4, 32)
        out = bl(x)
        assert out.shape == (4, 6)

    def test_scale_parameter_exists(self):
        bl = BinaryLinear(32, 6)
        assert isinstance(bl.scale, torch.nn.Parameter)
        assert bl.scale.shape == (6,)

    def test_gradients_flow_through_ste(self):
        bl = BinaryLinear(32, 6)
        x = torch.randn(4, 32)
        out = bl(x)
        loss = out.sum()
        loss.backward()
        assert bl.linear.weight.grad is not None
        assert bl.scale.grad is not None

    def test_export_binary_keys(self):
        bl = BinaryLinear(32, 6)
        exported = bl.export_binary()
        expected_keys = {"packed_weights", "scale", "bias", "in_features", "out_features"}
        assert set(exported.keys()) == expected_keys

    def test_export_binary_packed_dtype(self):
        bl = BinaryLinear(32, 6)
        exported = bl.export_binary()
        assert exported["packed_weights"].dtype == torch.uint8

    def test_export_binary_shape(self):
        bl = BinaryLinear(32, 6)
        exported = bl.export_binary()
        assert exported["packed_weights"].shape == (6, 32)
        assert exported["in_features"] == 32
        assert exported["out_features"] == 6


class TestBinaryClassifier:

    def test_forward_shape(self, binary_classifier):
        x = torch.randn(4, 32)
        out = binary_classifier(x)
        assert out.shape == (4, 6)

    def test_includes_batchnorm(self, binary_classifier):
        assert isinstance(binary_classifier.bn, torch.nn.BatchNorm1d)

    def test_includes_binary_linear(self, binary_classifier):
        assert isinstance(binary_classifier.head, BinaryLinear)
