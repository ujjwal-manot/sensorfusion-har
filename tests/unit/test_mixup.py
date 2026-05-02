import torch
import pytest
from model.mixup import reservoir_manifold_mixup


class TestReservoirManifoldMixup:

    def test_returns_scalar(self, model, synthetic_batch, synthetic_labels):
        x1, x2 = synthetic_batch[:4], synthetic_batch[4:]
        y1, y2 = synthetic_labels[:4], synthetic_labels[4:]
        loss = reservoir_manifold_mixup(model, x1, x2, y1, y2, torch.nn.CrossEntropyLoss(), alpha=0.2)
        assert loss.shape == ()

    def test_positive_loss(self, model, synthetic_batch, synthetic_labels):
        x1, x2 = synthetic_batch[:4], synthetic_batch[4:]
        y1, y2 = synthetic_labels[:4], synthetic_labels[4:]
        loss = reservoir_manifold_mixup(model, x1, x2, y1, y2, torch.nn.CrossEntropyLoss(), alpha=0.2)
        assert loss.item() > 0

    def test_gradient_flows(self, model, synthetic_batch, synthetic_labels):
        x1, x2 = synthetic_batch[:4], synthetic_batch[4:]
        y1, y2 = synthetic_labels[:4], synthetic_labels[4:]
        loss = reservoir_manifold_mixup(model, x1, x2, y1, y2, torch.nn.CrossEntropyLoss(), alpha=0.2)
        loss.backward()
        grad_found = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert grad_found
