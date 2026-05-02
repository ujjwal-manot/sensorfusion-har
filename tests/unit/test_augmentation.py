import numpy as np
import torch
import pytest
from model.augmentation import SensorAugmentor, AugmentedDataset


class TestSensorAugmentor:

    @pytest.fixture
    def augmentor(self):
        return SensorAugmentor(p=1.0)

    @pytest.fixture
    def sample_np(self):
        return np.random.randn(128, 6).astype(np.float32)

    @pytest.fixture
    def sample_tensor(self):
        return torch.randn(128, 6)

    def test_jitter_shape(self, augmentor, sample_np):
        out = augmentor.jitter(sample_np)
        assert out.shape == sample_np.shape

    def test_jitter_adds_noise(self, augmentor, sample_np):
        out = augmentor.jitter(sample_np)
        assert not np.allclose(out, sample_np)

    def test_scaling_shape(self, augmentor, sample_np):
        out = augmentor.scaling(sample_np)
        assert out.shape == sample_np.shape

    def test_rotation_shape(self, augmentor, sample_np):
        out = augmentor.rotation(sample_np)
        assert out.shape == sample_np.shape

    def test_permutation_shape(self, augmentor, sample_np):
        out = augmentor.permutation(sample_np)
        assert out.shape == sample_np.shape

    def test_time_warp_shape(self, augmentor, sample_np):
        out = augmentor.time_warp(sample_np)
        assert out.shape == sample_np.shape

    def test_magnitude_warp_shape(self, augmentor, sample_np):
        out = augmentor.magnitude_warp(sample_np)
        assert out.shape == sample_np.shape

    def test_channel_dropout_shape(self, augmentor, sample_np):
        out = augmentor.channel_dropout(sample_np, p_drop=0.5)
        assert out.shape == sample_np.shape

    def test_channel_dropout_zeros_channels(self, augmentor, sample_np):
        out = augmentor.channel_dropout(sample_np, p_drop=0.99)
        zero_channels = (out == 0).all(axis=0).sum()
        assert zero_channels >= 1

    def test_call_numpy(self, augmentor, sample_np):
        out = augmentor(sample_np)
        assert isinstance(out, np.ndarray)
        assert out.shape == sample_np.shape

    def test_call_tensor(self, augmentor, sample_tensor):
        out = augmentor(sample_tensor)
        assert isinstance(out, torch.Tensor)
        assert out.shape == sample_tensor.shape

    def test_augment_batch_numpy(self, augmentor):
        batch = np.random.randn(4, 128, 6).astype(np.float32)
        out = augmentor.augment_batch(batch)
        assert isinstance(out, np.ndarray)
        assert out.shape == batch.shape

    def test_augment_batch_tensor(self, augmentor):
        batch = torch.randn(4, 128, 6)
        out = augmentor.augment_batch(batch)
        assert isinstance(out, torch.Tensor)
        assert out.shape == batch.shape

    def test_probability_zero_noop(self, sample_np):
        aug = SensorAugmentor(p=0.0)
        out = aug(sample_np)
        np.testing.assert_array_equal(out, sample_np)


class TestAugmentedDataset:

    def test_len(self, synthetic_dataset):
        aug_ds = AugmentedDataset(synthetic_dataset)
        assert len(aug_ds) == len(synthetic_dataset)

    def test_getitem_returns_tuple(self, synthetic_dataset):
        aug_ds = AugmentedDataset(synthetic_dataset)
        x, y = aug_ds[0]
        assert isinstance(x, torch.Tensor)
