import numpy as np
import torch
from scipy.interpolate import CubicSpline


class SensorAugmentor:

    def __init__(self, p=0.5):
        self.p = p
        self.augmentations = [
            self.jitter,
            self.scaling,
            self.rotation,
            self.permutation,
            self.time_warp,
            self.magnitude_warp,
            self.channel_dropout,
        ]

    def jitter(self, x, sigma=0.05):
        return x + np.random.normal(0, sigma, x.shape)

    def scaling(self, x, sigma=0.1):
        factors = np.random.normal(1, sigma, (1, x.shape[1]))
        return x * factors

    def rotation(self, x):
        def _random_rotation_matrix():
            angle = np.random.uniform(-15, 15) * np.pi / 180.0
            axis = np.random.randn(3)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            return R

        result = x.copy()
        R_acc = _random_rotation_matrix()
        R_gyro = _random_rotation_matrix()
        result[:, :3] = result[:, :3] @ R_acc.T
        result[:, 3:6] = result[:, 3:6] @ R_gyro.T
        return result

    def permutation(self, x, max_segments=5):
        num_segments = np.random.randint(2, max_segments + 1)
        points = np.sort(np.random.choice(range(1, x.shape[0]), num_segments - 1, replace=False))
        segments = np.split(x, points, axis=0)
        np.random.shuffle(segments)
        return np.concatenate(segments, axis=0)

    def time_warp(self, x, sigma=0.2, knots=4):
        seq_len = x.shape[0]
        orig_steps = np.arange(seq_len)
        random_knot_x = np.linspace(0, seq_len - 1, knots + 2)
        random_knot_y = np.random.normal(1.0, sigma, knots + 2)
        random_knot_y[0] = 1.0
        random_knot_y[-1] = 1.0
        spline = CubicSpline(random_knot_x, random_knot_y)
        warp_factors = spline(orig_steps)
        warp_factors = np.maximum(warp_factors, 0.1)
        cumulative = np.cumsum(warp_factors)
        cumulative = (cumulative - cumulative[0]) / (cumulative[-1] - cumulative[0]) * (seq_len - 1)
        result = np.zeros_like(x)
        for ch in range(x.shape[1]):
            result[:, ch] = np.interp(orig_steps, cumulative, x[:, ch])
        return result

    def magnitude_warp(self, x, sigma=0.2, knots=4):
        seq_len = x.shape[0]
        random_knot_x = np.linspace(0, seq_len - 1, knots + 2)
        random_knot_y = np.random.normal(1.0, sigma, knots + 2)
        spline = CubicSpline(random_knot_x, random_knot_y)
        warp_curve = spline(np.arange(seq_len)).reshape(-1, 1)
        return x * warp_curve

    def channel_dropout(self, x, p_drop=0.1):
        mask = np.random.binomial(1, 1 - p_drop, (1, x.shape[1])).astype(x.dtype)
        return x * mask

    def __call__(self, x):
        is_tensor = isinstance(x, torch.Tensor)
        device = x.device if is_tensor else None
        dtype = x.dtype if is_tensor else None
        x_np = x.cpu().numpy().copy() if is_tensor else x.copy()

        for aug in self.augmentations:
            if np.random.random() < self.p:
                x_np = aug(x_np)

        if is_tensor:
            return torch.from_numpy(x_np.copy()).to(dtype=dtype, device=device)
        return x_np

    def augment_batch(self, batch):
        is_tensor = isinstance(batch, torch.Tensor)
        device = batch.device if is_tensor else None
        dtype = batch.dtype if is_tensor else None

        if is_tensor:
            batch_np = batch.cpu().numpy()
        else:
            batch_np = batch

        results = []
        for i in range(batch_np.shape[0]):
            results.append(self(batch_np[i]))

        if is_tensor:
            return torch.stack([torch.from_numpy(r) if isinstance(r, np.ndarray) else r for r in results]).to(dtype=dtype, device=device)
        return np.stack(results)


class AugmentedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, augmentor=None):
        self.dataset = dataset
        self.augmentor = augmentor if augmentor is not None else SensorAugmentor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.augmentor(x)
        return x, y
