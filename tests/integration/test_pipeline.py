"""Integration tests for the full SensorFusion-HAR pipeline."""
import torch
import pytest
from model.sensorfusion import SensorFusionHAR
from model.contrastive import SensorSimCLR, pretrain_contrastive, transfer_weights
from model.masked_pretrain import MaskedSensorModel, masked_pretrain, transfer_masked_weights
from model.multitask import MultiTaskHAR, SubjectLabeledDataset, train_multitask
from model.curriculum import CurriculumScheduler, CurriculumTrainer
from model.adversarial import evaluate_adversarial_robustness
from model.drift import evaluate_drift_robustness
from model.transitions import evaluate_transition_accuracy
from model.energy import count_macs, estimate_energy, compare_models_energy
from model.augmentation import SensorAugmentor
from model.personalization import few_shot_personalize


class TestPretrainTransferPipeline:

    @pytest.fixture
    def dataset(self):
        X = torch.randn(64, 128, 6)
        y = torch.randint(0, 6, (64,))
        return torch.utils.data.TensorDataset(X, y)

    def test_simclr_pretrain_then_transfer(self, dataset, device):
        simclr = SensorSimCLR(6, 32)
        augmentor = SensorAugmentor()
        pretrain_contrastive(simclr, dataset, augmentor, device, epochs=2, batch_size=16)
        target = SensorFusionHAR(6, 32, 6)
        transferred = transfer_weights(simclr, target)
        out = transferred(torch.randn(4, 128, 6))
        assert out.shape == (4, 6)

    def test_masked_pretrain_then_transfer(self, dataset, device):
        msm = MaskedSensorModel(6, 32)
        masked_pretrain(msm, dataset, device, epochs=2, batch_size=16)
        target = SensorFusionHAR(6, 32, 6)
        transferred = transfer_masked_weights(msm, target)
        out = transferred(torch.randn(4, 128, 6))
        assert out.shape == (4, 6)


class TestCurriculumPipeline:

    def test_full_curriculum_training(self, device):
        model = SensorFusionHAR(6, 32, 6)
        X_train = torch.randn(32, 128, 6)
        y_train = torch.tensor([3, 4, 5, 0, 1, 2] * 5 + [3, 4], dtype=torch.long)
        X_test = torch.randn(12, 128, 6)
        y_test = torch.randint(0, 6, (12,))
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        test_ds = torch.utils.data.TensorDataset(X_test, y_test)

        sched = CurriculumScheduler("ucihar", total_epochs=6, num_phases=3)
        trainer = CurriculumTrainer(model, train_ds, test_ds, device, sched, batch_size=8)
        trained_model, history = trainer.train(epochs=6)

        out = trained_model(torch.randn(2, 128, 6))
        assert out.shape == (2, 6)
        assert len(history["train_loss"]) == 6


class TestMultitaskPipeline:

    def test_multitask_then_extract_backbone(self, device):
        X = torch.randn(64, 128, 6)
        y = torch.randint(0, 6, (64,))
        subjects = torch.randint(0, 4, (64,))
        train_ds = SubjectLabeledDataset(X, y, subjects)

        X_test = torch.randn(16, 128, 6)
        y_test = torch.randint(0, 6, (16,))
        s_test = torch.randint(0, 4, (16,))
        test_ds = SubjectLabeledDataset(X_test, y_test, s_test)

        mt = MultiTaskHAR(6, 32, 6, num_subjects=4)
        trained = train_multitask(mt, train_ds, test_ds, device, epochs=2, batch_size=16)
        backbone = trained.extract_backbone()
        out = backbone(torch.randn(4, 128, 6))
        assert out.shape == (4, 6)


class TestEvaluationPipeline:

    @pytest.fixture
    def trained_setup(self, device):
        model = SensorFusionHAR(6, 32, 6)
        X = torch.randn(16, 128, 6)
        y = torch.randint(0, 6, (16,))
        ds = torch.utils.data.TensorDataset(X, y)
        return model, ds, device

    def test_adversarial_eval(self, trained_setup):
        model, ds, device = trained_setup
        results = evaluate_adversarial_robustness(
            model, ds, device, epsilons=[0, 0.05], attack="fgsm"
        )
        assert len(results["fgsm_accuracy"]) == 2

    def test_drift_eval(self, trained_setup):
        model, ds, device = trained_setup
        results = evaluate_drift_robustness(
            model, ds, device,
            drift_types=["bias"],
            drift_levels={"bias": [0, 0.01]},
        )
        assert "bias_accuracy" in results

    def test_transition_eval(self, trained_setup):
        model, ds, device = trained_setup
        results = evaluate_transition_accuracy(model, ds, device)
        assert "stable_accuracy" in results

    def test_energy_profiling(self, trained_setup):
        model, _, _ = trained_setup
        macs = count_macs(model)
        energy = estimate_energy(macs, "fp32")
        comparison = compare_models_energy({"model": model})
        assert macs["total"] > 0
        assert energy["total_joules"] > 0
        assert comparison["model"]["total_macs"] > 0

    def test_personalization(self, trained_setup):
        model, _, device = trained_setup
        sx = torch.randn(30, 128, 6)
        sy = torch.randint(0, 6, (30,))
        personalized = few_shot_personalize(model, sx, sy, device, k_shots=5, steps=3)
        out = personalized(torch.randn(4, 128, 6))
        assert out.shape == (4, 6)
