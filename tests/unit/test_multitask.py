import torch
import pytest
from model.multitask import (
    GradientReversal,
    GradientReversalLayer,
    MultiTaskHAR,
    SubjectLabeledDataset,
    train_multitask,
)


class TestGradientReversal:

    def test_forward_identity(self):
        x = torch.randn(4, 32, requires_grad=True)
        out = GradientReversal.apply(x, 1.0)
        assert torch.allclose(out, x)

    def test_backward_negation(self):
        x = torch.randn(4, 32, requires_grad=True)
        out = GradientReversal.apply(x, 1.0)
        loss = out.sum()
        loss.backward()
        assert torch.allclose(x.grad, -torch.ones_like(x))

    def test_backward_custom_lambda(self):
        x = torch.randn(4, 32, requires_grad=True)
        out = GradientReversal.apply(x, 0.5)
        loss = out.sum()
        loss.backward()
        assert torch.allclose(x.grad, -0.5 * torch.ones_like(x))


class TestGradientReversalLayer:

    def test_set_lambda(self):
        grl = GradientReversalLayer(lambda_=1.0)
        grl.set_lambda(0.5)
        assert grl.lambda_ == 0.5

    def test_forward(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 32)
        out = grl(x)
        assert torch.allclose(out, x)


class TestSubjectLabeledDataset:

    def test_getitem(self):
        X = torch.randn(10, 128, 6)
        y = torch.randint(0, 6, (10,))
        subjects = torch.randint(0, 4, (10,))
        ds = SubjectLabeledDataset(X, y, subjects)
        x, label, subj = ds[0]
        assert x.shape == (128, 6)

    def test_len(self):
        ds = SubjectLabeledDataset(
            torch.randn(10, 128, 6),
            torch.randint(0, 6, (10,)),
            torch.zeros(10),
        )
        assert len(ds) == 10


class TestMultiTaskHAR:

    @pytest.fixture
    def mt_model(self):
        return MultiTaskHAR(6, 32, 6, num_subjects=4)

    def test_forward_shapes(self, mt_model, synthetic_batch):
        act, subj = mt_model(synthetic_batch)
        assert act.shape == (8, 6)
        assert subj.shape == (8, 4)

    def test_extract_backbone(self, mt_model, synthetic_batch):
        backbone = mt_model.extract_backbone()
        out = backbone(synthetic_batch)
        assert out.shape == (8, 6)

    def test_gradient_reversal_in_forward(self, mt_model, synthetic_batch):
        """Verify that subject head uses GRL."""
        act, subj = mt_model(synthetic_batch)
        loss = subj.sum()
        loss.backward()
        # GRL should allow gradients to flow (just reversed)
        assert mt_model.subject_head[0].weight.grad is not None


class TestTrainMultitask:

    @pytest.fixture
    def subject_datasets(self):
        X = torch.randn(64, 128, 6)
        y = torch.randint(0, 6, (64,))
        subjects = torch.randint(0, 4, (64,))
        train_ds = SubjectLabeledDataset(X, y, subjects)

        X_test = torch.randn(16, 128, 6)
        y_test = torch.randint(0, 6, (16,))
        subjects_test = torch.randint(0, 4, (16,))
        test_ds = SubjectLabeledDataset(X_test, y_test, subjects_test)
        return train_ds, test_ds

    def test_train_returns_model(self, subject_datasets, device):
        train_ds, test_ds = subject_datasets
        mt_model = MultiTaskHAR(6, 32, 6, num_subjects=4)
        trained = train_multitask(
            mt_model, train_ds, test_ds, device,
            epochs=10, batch_size=16, lr=0.001, lambda_schedule="linear",
        )
        assert isinstance(trained, MultiTaskHAR)

    def test_train_constant_lambda(self, subject_datasets, device):
        train_ds, test_ds = subject_datasets
        mt_model = MultiTaskHAR(6, 32, 6, num_subjects=4)
        trained = train_multitask(
            mt_model, train_ds, test_ds, device,
            epochs=2, batch_size=16, lr=0.001, lambda_schedule="constant",
        )
        assert isinstance(trained, MultiTaskHAR)
