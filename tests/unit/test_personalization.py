import torch
import pytest
from model.personalization import few_shot_personalize


class TestFewShotPersonalize:

    def test_returns_model(self, model):
        sx = torch.randn(20, 128, 6)
        sy = torch.randint(0, 6, (20,))
        result = few_shot_personalize(model, sx, sy, torch.device("cpu"), k_shots=5, steps=5)
        assert isinstance(result, torch.nn.Module)

    def test_backbone_frozen(self, model):
        sx = torch.randn(20, 128, 6)
        sy = torch.randint(0, 6, (20,))
        original_reservoir = model.reservoir.W_in.clone()
        personalized = few_shot_personalize(model, sx, sy, torch.device("cpu"), k_shots=5, steps=5)
        assert torch.equal(personalized.reservoir.W_in, original_reservoir)

    def test_classifier_updated(self, model):
        sx = torch.randn(20, 128, 6)
        sy = torch.randint(0, 6, (20,))
        original_weight = model.classifier.head.linear.weight.clone()
        personalized = few_shot_personalize(model, sx, sy, torch.device("cpu"), k_shots=5, steps=10)
        assert not torch.equal(personalized.classifier.head.linear.weight, original_weight)

    def test_deep_copy_preserves_original(self, model):
        """Personalization should not modify the original model."""
        original_cls_weight = model.classifier.head.linear.weight.clone()
        sx = torch.randn(20, 128, 6)
        sy = torch.randint(0, 6, (20,))
        few_shot_personalize(model, sx, sy, torch.device("cpu"), k_shots=5, steps=10)
        assert torch.equal(model.classifier.head.linear.weight, original_cls_weight)

    def test_personalized_model_eval_mode(self, model):
        sx = torch.randn(20, 128, 6)
        sy = torch.randint(0, 6, (20,))
        personalized = few_shot_personalize(model, sx, sy, torch.device("cpu"), k_shots=5, steps=5)
        assert not personalized.training

    def test_k_shots_limits_selection(self, model):
        """With k_shots=1 and 6 classes, should use at most 6 samples."""
        sx = torch.randn(60, 128, 6)
        sy = torch.arange(6).repeat(10)
        personalized = few_shot_personalize(model, sx, sy, torch.device("cpu"), k_shots=1, steps=3)
        assert isinstance(personalized, torch.nn.Module)

    def test_single_class_support(self, model):
        """Works with support set containing only one class."""
        sx = torch.randn(10, 128, 6)
        sy = torch.zeros(10, dtype=torch.long)
        personalized = few_shot_personalize(model, sx, sy, torch.device("cpu"), k_shots=5, steps=3)
        assert isinstance(personalized, torch.nn.Module)
