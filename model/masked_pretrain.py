import torch
import torch.nn as nn

from .reservoir import EchoStateNetwork
from .dsconv import DSConvEncoder
from .attention import PatchMicroAttention


class MaskedSensorModel(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, mask_ratio=0.15):
        super().__init__()
        self.input_channels = input_channels
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_channels))

        self.backbone_reservoir = EchoStateNetwork(input_channels, reservoir_size)
        self.backbone_dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.backbone_attention = PatchMicroAttention(
            in_channels=48, seq_len=32, d_model=32, ff_dim=48
        )

        self.reconstruction_head = nn.Linear(32, input_channels)

    def forward(self, x, mask=None):
        batch_size, seq_len, channels = x.shape

        if mask is None:
            mask = create_mask(batch_size, seq_len, self.mask_ratio, x.device)

        mask_expanded = mask.unsqueeze(-1).float()
        x_masked = x * (1.0 - mask_expanded) + self.mask_token * mask_expanded

        h = self.backbone_reservoir(x_masked)
        h = h.transpose(1, 2)
        h = self.backbone_dsconv(h)
        h = self.backbone_attention(h)

        reconstruction = self.reconstruction_head(h)
        reconstruction = reconstruction.unsqueeze(1).expand(-1, seq_len, -1)

        return reconstruction, mask


def create_mask(batch_size, seq_len, mask_ratio, device):
    num_masked = max(1, int(seq_len * mask_ratio))
    mask = torch.zeros(batch_size, seq_len, device=device)
    for i in range(batch_size):
        indices = torch.randperm(seq_len, device=device)[:num_masked]
        mask[i, indices] = 1.0
    return mask


def masked_pretrain(model, dataset, device, epochs=50, batch_size=128, lr=0.0003, mask_ratio=0.15):
    model = model.to(device)
    model.train()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for x, _ in loader:
            x = x.to(device)
            batch_size_actual, seq_len, channels = x.shape

            mask = create_mask(batch_size_actual, seq_len, mask_ratio, device)

            reconstruction, mask = model(x, mask=mask)

            mask_expanded = mask.unsqueeze(-1).float()
            masked_recon = reconstruction * mask_expanded
            masked_target = x * mask_expanded
            num_masked_elements = mask_expanded.sum().clamp(min=1.0)
            loss = ((masked_recon - masked_target) ** 2).sum() / (num_masked_elements * channels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch [{epoch + 1}/{epochs}] MSM Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

    return model


def transfer_masked_weights(pretrained_msm, target_model):
    target_model.reservoir.load_state_dict(pretrained_msm.backbone_reservoir.state_dict())
    target_model.dsconv.load_state_dict(pretrained_msm.backbone_dsconv.state_dict())
    target_model.attention.load_state_dict(pretrained_msm.backbone_attention.state_dict())
    return target_model
