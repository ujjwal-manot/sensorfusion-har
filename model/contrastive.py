import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .reservoir import EchoStateNetwork
from .dsconv import DSConvEncoder
from .attention import PatchMicroAttention


class SensorSimCLR(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32):
        super().__init__()
        self.reservoir = EchoStateNetwork(input_channels, reservoir_size)
        self.dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.projection = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        x = self.reservoir(x)
        x = x.transpose(1, 2)
        x = self.dsconv(x)
        x = self.attention(x)
        x = self.projection(x)
        return F.normalize(x, dim=1)

    def get_features(self, x):
        x = self.reservoir(x)
        x = x.transpose(1, 2)
        x = self.dsconv(x)
        x = self.attention(x)
        return x


def nt_xent_loss(z1, z2, temperature=0.1):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)

    pos_idx_top = torch.arange(batch_size, device=z.device) + batch_size
    pos_idx_bottom = torch.arange(batch_size, device=z.device)
    labels = torch.cat([pos_idx_top, pos_idx_bottom], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss


def pretrain_contrastive(model, dataset, augmentor, device, epochs=50, batch_size=128, lr=0.0003, temperature=0.1):
    model = model.to(device)
    model.train()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for x, _ in loader:
            x = x.to(device)

            view1 = augmentor.augment_batch(x)
            view2 = augmentor.augment_batch(x)

            z1 = model(view1)
            z2 = model(view2)

            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

    return model


def transfer_weights(pretrained_simclr, target_model):
    target_model.reservoir.load_state_dict(pretrained_simclr.reservoir.state_dict())
    target_model.dsconv.load_state_dict(pretrained_simclr.dsconv.state_dict())
    target_model.attention.load_state_dict(pretrained_simclr.attention.state_dict())
    return target_model
