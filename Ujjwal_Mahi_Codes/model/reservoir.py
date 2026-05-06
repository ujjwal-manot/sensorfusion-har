import math

import torch
import torch.nn as nn

__all__ = ["EchoStateNetwork"]


class EchoStateNetwork(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, spectral_radius=0.9,
                 sparsity=0.8, learnable_sr=False, use_diff_states=False,
                 reservoir_dropout=0.0):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.use_diff_states = use_diff_states
        self.reservoir_dropout = reservoir_dropout
        self._output_dim = reservoir_size * 2 if use_diff_states else reservoir_size

        W_in = torch.randn(input_channels, reservoir_size) * 0.1
        self.register_buffer("W_in", W_in)

        W_res = torch.randn(reservoir_size, reservoir_size)
        mask = (torch.rand(reservoir_size, reservoir_size) > sparsity).float()
        W_res = W_res * mask
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        current_radius = eigenvalues.max().item()
        if current_radius > 0:
            W_res = W_res * (spectral_radius / current_radius)
        self.register_buffer("W_res", W_res)
        self.register_buffer("_base_sr", torch.tensor(float(spectral_radius)))

        if learnable_sr:
            init_logit = math.log(spectral_radius / (1.0 - spectral_radius + 1e-7))
            self.sr_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.sr_logit = None

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def effective_spectral_radius(self):
        if self.sr_logit is not None:
            return torch.sigmoid(self.sr_logit)
        return self._base_sr

    def _scaled_reservoir_weights(self):
        if self.sr_logit is not None:
            sr = self.effective_spectral_radius
            return self.W_res * (sr / (self._base_sr + 1e-7))
        return self.W_res

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.reservoir_size, device=x.device, dtype=x.dtype)
        W_r = self._scaled_reservoir_weights()
        x_proj = x @ self.W_in

        if self.training and self.reservoir_dropout > 0:
            drop_mask = (torch.rand(batch, self.reservoir_size, device=x.device) > self.reservoir_dropout).float()
            drop_mask = drop_mask / (1.0 - self.reservoir_dropout)
        else:
            drop_mask = None

        states = []
        for t in range(seq_len):
            h = torch.tanh(x_proj[:, t] + h @ W_r)
            if drop_mask is not None:
                h = h * drop_mask
            states.append(h.unsqueeze(1))

        h_seq = torch.cat(states, dim=1)

        if self.use_diff_states:
            h_prev = torch.cat([torch.zeros_like(h_seq[:, :1]), h_seq[:, :-1]], dim=1)
            delta_h = h_seq - h_prev
            return torch.cat([h_seq, delta_h], dim=-1)

        return h_seq

    @classmethod
    def spectral_init(cls, dataset, input_channels=6, reservoir_size=32, spectral_radius=0.9, **kwargs):
        if hasattr(dataset, "X"):
            data = dataset.X
        else:
            samples = []
            for i in range(len(dataset)):
                item = dataset[i]
                samples.append(item[0] if isinstance(item, (tuple, list)) else item)
            data = torch.stack(samples)

        flat = data.reshape(-1, input_channels)
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        autocorr = (centered.T @ centered) / centered.shape[0]

        eigenvalues, eigenvectors = torch.linalg.eigh(autocorr)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idx]

        k = min(input_channels, reservoir_size)
        W_in = torch.zeros(input_channels, reservoir_size)
        W_in[:, :k] = eigenvectors[:, :k]
        if reservoir_size > k:
            W_in[:, k:] = torch.randn(input_channels, reservoir_size - k) * 0.01
        W_in = W_in * 0.1 / (W_in.norm(dim=0, keepdim=True).clamp(min=1e-8) / W_in.shape[0] ** 0.5)

        instance = cls(input_channels, reservoir_size, spectral_radius, **kwargs)
        instance.W_in.copy_(W_in)
        return instance
