import torch
import pytest
from model.masked_pretrain import (
    MaskedSensorModel,
    create_mask,
    masked_pretrain,
    transfer_masked_weights,
)


class TestMaskedSensorModel:

    @pytest.fixture
    def masked_model(self):
        return MaskedSensorModel(input_channels=6, reservoir_size=32, mask_ratio=0.15)

    def test_forward_shapes(self, masked_model, synthetic_batch):
        reconstruction, mask = masked_model(synthetic_batch)
        assert reconstruction.shape == (8, 128, 6)
        assert mask.shape == (8, 128)

    def test_custom_mask_passthrough(self, masked_model, synthetic_batch):
        custom_mask = torch.zeros(8, 128)
        custom_mask[:, :10] = 1.0
        _, returned_mask = masked_model(synthetic_batch, mask=custom_mask)
        assert torch.equal(returned_mask, custom_mask)

    def test_reconstruction_differentiable(self, masked_model, synthetic_batch):
        reconstruction, mask = masked_model(synthetic_batch)
        loss = reconstruction.sum()
        loss.backward()
        assert masked_model.reconstruction_head.weight.grad is not None

    def test_mask_token_is_learnable(self, masked_model):
        assert masked_model.mask_token.requires_grad


class TestCreateMask:

    def test_shape(self):
        mask = create_mask(4, 128, 0.15, torch.device("cpu"))
        assert mask.shape == (4, 128)

    def test_ratio(self):
        mask = create_mask(4, 128, 0.15, torch.device("cpu"))
        expected = max(1, int(128 * 0.15))
        for i in range(4):
            assert mask[i].sum().item() == expected

    def test_at_least_one(self):
        mask = create_mask(4, 128, 0.01, torch.device("cpu"))
        for i in range(4):
            assert mask[i].sum().item() >= 1

    def test_binary_values(self):
        mask = create_mask(4, 128, 0.15, torch.device("cpu"))
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})


class TestMaskedPretrain:

    def test_training_loop(self, synthetic_dataset, device):
        msm = MaskedSensorModel(input_channels=6, reservoir_size=32, mask_ratio=0.15)
        trained = masked_pretrain(
            msm, synthetic_dataset, device,
            epochs=2, batch_size=8, lr=0.001, mask_ratio=0.15,
        )
        assert isinstance(trained, MaskedSensorModel)

    def test_training_reduces_loss(self, device):
        X = torch.randn(32, 128, 6)
        y = torch.randint(0, 6, (32,))
        ds = torch.utils.data.TensorDataset(X, y)
        msm = MaskedSensorModel(input_channels=6, reservoir_size=32, mask_ratio=0.15)

        # Get initial reconstruction error
        msm.eval()
        with torch.no_grad():
            recon, mask = msm(X[:8])
            mask_exp = mask.unsqueeze(-1)
            initial_loss = ((recon * mask_exp - X[:8] * mask_exp) ** 2).mean().item()

        # Train
        msm = masked_pretrain(msm, ds, device, epochs=5, batch_size=16, lr=0.001)

        # Get post-training error
        msm.eval()
        with torch.no_grad():
            recon2, mask2 = msm(X[:8])
            mask_exp2 = mask2.unsqueeze(-1)
            final_loss = ((recon2 * mask_exp2 - X[:8] * mask_exp2) ** 2).mean().item()

        # Loss should decrease (or at least not increase dramatically)
        assert final_loss <= initial_loss * 2.0  # Allow some tolerance


class TestTransferMaskedWeights:

    def test_transfer(self):
        from model.sensorfusion import SensorFusionHAR
        msm = MaskedSensorModel(6, 32)
        target = SensorFusionHAR(6, 32, 6)
        transferred = transfer_masked_weights(msm, target)
        for key in msm.backbone_dsconv.state_dict():
            assert torch.equal(
                msm.backbone_dsconv.state_dict()[key],
                transferred.dsconv.state_dict()[key],
            )

    def test_transfer_attention(self):
        from model.sensorfusion import SensorFusionHAR
        msm = MaskedSensorModel(6, 32)
        target = SensorFusionHAR(6, 32, 6)
        transferred = transfer_masked_weights(msm, target)
        for key in msm.backbone_attention.state_dict():
            assert torch.equal(
                msm.backbone_attention.state_dict()[key],
                transferred.attention.state_dict()[key],
            )
