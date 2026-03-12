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
