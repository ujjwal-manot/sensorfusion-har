import torch
import pytest
from model.contrastive import SensorSimCLR, nt_xent_loss, pretrain_contrastive, transfer_weights
from model.augmentation import SensorAugmentor


class TestSensorSimCLR:

    @pytest.fixture
    def simclr(self):
        return SensorSimCLR(input_channels=6, reservoir_size=32)

    def test_forward_shape(self, simclr, synthetic_batch):
        out = simclr(synthetic_batch)
        assert out.shape == (8, 32)

    def test_output_normalized(self, simclr, synthetic_batch):
        out = simclr(synthetic_batch)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_get_features_shape(self, simclr, synthetic_batch):
        feat = simclr.get_features(synthetic_batch)
        assert feat.shape == (8, 32)

    def test_features_differ_from_projection(self, simclr, synthetic_batch):
        feat = simclr.get_features(synthetic_batch)
        proj = simclr(synthetic_batch)
        assert not torch.allclose(feat, proj)


class TestNtXentLoss:

    def test_scalar(self):
        z1 = torch.randn(4, 32)
        z2 = torch.randn(4, 32)
        loss = nt_xent_loss(z1, z2)
        assert loss.shape == ()

    def test_positive(self):
        z1 = torch.randn(4, 32)
        z2 = torch.randn(4, 32)
        loss = nt_xent_loss(z1, z2)
        assert loss.item() > 0

    def test_identical_views_low_loss(self):
        z = torch.randn(4, 32)
        z = torch.nn.functional.normalize(z, dim=1)
        loss = nt_xent_loss(z, z)
        # Same views should have relatively low loss
        random_loss = nt_xent_loss(torch.randn(4, 32), torch.randn(4, 32))
        assert loss.item() < random_loss.item()

    def test_temperature_effect(self):
        z1 = torch.randn(4, 32)
        z2 = torch.randn(4, 32)
        loss_low_t = nt_xent_loss(z1, z2, temperature=0.05)
        loss_high_t = nt_xent_loss(z1, z2, temperature=1.0)
        # Lower temperature → sharper distribution → typically different loss
        assert loss_low_t.item() != loss_high_t.item()


class TestPretrainContrastive:

    def test_training_loop(self, synthetic_dataset, device):
        simclr = SensorSimCLR(input_channels=6, reservoir_size=32)
        augmentor = SensorAugmentor()
        trained = pretrain_contrastive(
            simclr, synthetic_dataset, augmentor, device,
            epochs=2, batch_size=8, lr=0.001, temperature=0.1,
        )
        assert isinstance(trained, SensorSimCLR)

    def test_weights_change(self, device):
        simclr = SensorSimCLR(input_channels=6, reservoir_size=32)
        original_weight = simclr.projection[0].weight.clone()
        augmentor = SensorAugmentor()

        X = torch.randn(32, 128, 6)
        y = torch.randint(0, 6, (32,))
        ds = torch.utils.data.TensorDataset(X, y)

        pretrain_contrastive(simclr, ds, augmentor, device, epochs=3, batch_size=16, lr=0.01)
        assert not torch.equal(simclr.projection[0].weight, original_weight)


class TestTransferWeights:

    def test_transfer_copies_state(self):
        from model.sensorfusion import SensorFusionHAR
        simclr = SensorSimCLR(6, 32)
        target = SensorFusionHAR(6, 32, 6)
        transferred = transfer_weights(simclr, target)
        for key in simclr.dsconv.state_dict():
            assert torch.equal(
                simclr.dsconv.state_dict()[key],
                transferred.dsconv.state_dict()[key],
            )

    def test_transfer_attention_weights(self):
        from model.sensorfusion import SensorFusionHAR
        simclr = SensorSimCLR(6, 32)
        target = SensorFusionHAR(6, 32, 6)
        transferred = transfer_weights(simclr, target)
        for key in simclr.attention.state_dict():
            assert torch.equal(
                simclr.attention.state_dict()[key],
                transferred.attention.state_dict()[key],
            )
