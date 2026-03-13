import torch
import torch.nn as nn

__all__ = ["EchoStateNetwork"]


class EchoStateNetwork(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, spectral_radius=0.9, sparsity=0.8):
        super().__init__()
        self.reservoir_size = reservoir_size

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

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.reservoir_size, device=x.device, dtype=x.dtype)
        states = []
        for t in range(seq_len):
            h = torch.tanh(x[:, t] @ self.W_in + h @ self.W_res)
            states.append(h.unsqueeze(1))
        return torch.cat(states, dim=1)

    @classmethod
    def spectral_init(cls, dataset, input_channels=6, reservoir_size=32, spectral_radius=0.9):
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

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.reservoir_size = reservoir_size

        instance.register_buffer("W_in", W_in)

        W_res = torch.randn(reservoir_size, reservoir_size)
        sparsity = 0.8
        mask = (torch.rand(reservoir_size, reservoir_size) > sparsity).float()
        W_res = W_res * mask
        eigs = torch.linalg.eigvals(W_res).abs()
        current_radius = eigs.max().item()
        if current_radius > 0:
            W_res = W_res * (spectral_radius / current_radius)
        instance.register_buffer("W_res", W_res)

        return instance
