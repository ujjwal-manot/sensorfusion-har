import torch
import pytest
from model.adversarial import fgsm_attack, pgd_attack, evaluate_adversarial_robustness


class TestFGSM:

    def test_output_shape(self, model, synthetic_batch, synthetic_labels):
        x_adv = fgsm_attack(model, synthetic_batch, synthetic_labels, epsilon=0.1, device=torch.device("cpu"))
        assert x_adv.shape == synthetic_batch.shape

    def test_perturbation_bounded(self, model, synthetic_batch, synthetic_labels):
        eps = 0.1
        x_adv = fgsm_attack(model, synthetic_batch, synthetic_labels, epsilon=eps, device=torch.device("cpu"))
        diff = (x_adv - synthetic_batch).abs()
        assert (diff <= eps + 1e-6).all()

    def test_modifies_input(self, model, synthetic_batch, synthetic_labels):
        x_adv = fgsm_attack(model, synthetic_batch, synthetic_labels, epsilon=0.1, device=torch.device("cpu"))
        assert not torch.allclose(x_adv, synthetic_batch)

    def test_zero_epsilon_noop(self, model, synthetic_batch, synthetic_labels):
        x_adv = fgsm_attack(model, synthetic_batch, synthetic_labels, epsilon=0.0, device=torch.device("cpu"))
        assert torch.allclose(x_adv, synthetic_batch, atol=1e-6)


class TestPGD:

    def test_output_shape(self, model, synthetic_batch, synthetic_labels):
        x_adv = pgd_attack(model, synthetic_batch, synthetic_labels, epsilon=0.1, alpha=0.025, num_steps=5, device=torch.device("cpu"))
        assert x_adv.shape == synthetic_batch.shape

    def test_perturbation_bounded(self, model, synthetic_batch, synthetic_labels):
        eps = 0.1
        x_adv = pgd_attack(model, synthetic_batch, synthetic_labels, epsilon=eps, alpha=0.025, num_steps=5, device=torch.device("cpu"))
        diff = (x_adv - synthetic_batch.to(x_adv.device)).abs()
        assert (diff <= eps + 1e-6).all()

    def test_modifies_input(self, model, synthetic_batch, synthetic_labels):
        x_adv = pgd_attack(model, synthetic_batch, synthetic_labels, epsilon=0.1, alpha=0.025, num_steps=3, device=torch.device("cpu"))
        assert not torch.allclose(x_adv, synthetic_batch)


class TestEvaluateAdversarialRobustness:

    @pytest.fixture
    def small_dataset(self):
        X = torch.randn(16, 128, 6)
        y = torch.randint(0, 6, (16,))
        return torch.utils.data.TensorDataset(X, y)

    def test_returns_expected_keys(self, model, small_dataset, device):
        results = evaluate_adversarial_robustness(
            model, small_dataset, device, epsilons=[0, 0.05], attack="both"
        )
        assert "epsilons" in results
        assert "clean_accuracy" in results
        assert "fgsm_accuracy" in results
        assert "pgd_accuracy" in results

    def test_clean_accuracy_first(self, model, small_dataset, device):
        results = evaluate_adversarial_robustness(
            model, small_dataset, device, epsilons=[0, 0.1], attack="both"
        )
        assert results["fgsm_accuracy"][0] == results["clean_accuracy"]
        assert results["pgd_accuracy"][0] == results["clean_accuracy"]

    def test_fgsm_only(self, model, small_dataset, device):
        results = evaluate_adversarial_robustness(
            model, small_dataset, device, epsilons=[0, 0.05], attack="fgsm"
        )
        assert len(results["fgsm_accuracy"]) == 2
        assert len(results["pgd_accuracy"]) == 0

    def test_pgd_only(self, model, small_dataset, device):
        results = evaluate_adversarial_robustness(
            model, small_dataset, device, epsilons=[0, 0.05], attack="pgd"
        )
        assert len(results["pgd_accuracy"]) == 2
        assert len(results["fgsm_accuracy"]) == 0

    def test_accuracy_in_range(self, model, small_dataset, device):
        results = evaluate_adversarial_robustness(
            model, small_dataset, device, epsilons=[0, 0.1], attack="both"
        )
        for acc in results["fgsm_accuracy"] + results["pgd_accuracy"]:
            assert 0.0 <= acc <= 1.0
