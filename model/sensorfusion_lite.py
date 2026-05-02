"""
SensorFusion-HAR Lite: Ablated architecture for ESP32 deployment.

Keeps: ESN (no diff states, no dropout) + DS-Conv (2 blocks, 32 ch)
Replaces: Spectral FFT Gate -> Simple FC Gate
          Patch MHA -> GAP + Dense
          Binary STE Head -> Standard Linear
          
Includes MaskedSensorModelLite for MSM pre-training on the Lite backbone.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SensorFusionLite",
    "MaskedSensorModelLite",
    "EchoStateNetworkLite",
    "DSConvEncoderLite",
    "SimpleFCGate",
    "transfer_masked_weights_lite",
]


# ---------------------------------------------------------------------------
# ESN Lite: no differential states, no stochastic masking
# ---------------------------------------------------------------------------
class EchoStateNetworkLite(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, spectral_radius=0.9,
                 sparsity=0.8, learnable_sr=True):
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
        self.register_buffer("_base_sr", torch.tensor(float(spectral_radius)))

        if learnable_sr:
            init_logit = math.log(spectral_radius / (1.0 - spectral_radius + 1e-7))
            self.sr_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.sr_logit = None

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

        states = []
        for t in range(seq_len):
            h = torch.tanh(x_proj[:, t] + h @ W_r)
            states.append(h.unsqueeze(1))

        return torch.cat(states, dim=1)  # (B, T, reservoir_size)


# ---------------------------------------------------------------------------
# DS-Conv Lite: 2 blocks, 32 channels
# ---------------------------------------------------------------------------
class DepthwiseSeparableBlockLite(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity="relu")

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class DSConvEncoderLite(nn.Module):

    def __init__(self, in_channels=32, out_channels=32):
        super().__init__()
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlockLite(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            DepthwiseSeparableBlockLite(out_channels, out_channels, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Simple FC Gate (replaces Spectral FFT Gate)
# ---------------------------------------------------------------------------
class SimpleFCGate(nn.Module):
    """Channel-wise gating via GAP + Linear + Sigmoid. No FFT."""

    def __init__(self, reservoir_dim, dsconv_channels, seq_len):
        super().__init__()
        self.channel_proj = nn.Conv1d(reservoir_dim, dsconv_channels, kernel_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(seq_len)
        self.gate_fc = nn.Linear(dsconv_channels, dsconv_channels)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, reservoir_out, dsconv_out):
        res_projected = self.channel_proj(reservoir_out)
        res_aligned = self.temporal_pool(res_projected)
        # GAP over time -> gate per channel
        gap = dsconv_out.mean(dim=2)  # (B, C)
        gate = torch.sigmoid(self.gate_fc(gap)).unsqueeze(2)  # (B, C, 1)
        return dsconv_out + gate * res_aligned


# ---------------------------------------------------------------------------
# SensorFusion-HAR Lite
# ---------------------------------------------------------------------------
class SensorFusionLite(nn.Module):
    """Ablated SensorFusion-HAR for ESP32 deployment.

    Architecture:
        Input (B, 50, 6)
        -> ESN Lite (B, 50, 32)
        -> transpose -> DS-Conv Lite (B, 32, 25)  [2 blocks, stride 1 then 2]
        -> Simple FC Gate (B, 32, 25)
        -> GAP (B, 32)
        -> Dense(32, 32) + ReLU (B, 32)
        -> BN + Linear(32, num_classes)
    """

    def __init__(self, input_channels=6, reservoir_size=32, num_classes=10):
        super().__init__()
        self.reservoir = EchoStateNetworkLite(
            input_channels, reservoir_size, learnable_sr=True
        )

        dsconv_out_channels = 32
        # After ESN: (B, T, 32) -> transpose -> (B, 32, T)
        # After DSConv with stride [1, 2]: T -> T//2
        # For T=50: seq_len after dsconv = 25
        dsconv_seq_len = 25  # 50 // 2

        self.dsconv = DSConvEncoderLite(
            in_channels=reservoir_size, out_channels=dsconv_out_channels
        )
        self.gate = SimpleFCGate(
            reservoir_dim=reservoir_size,
            dsconv_channels=dsconv_out_channels,
            seq_len=dsconv_seq_len,
        )

        # GAP + Dense replaces Patch MHA
        self.feature_dense = nn.Sequential(
            nn.Linear(dsconv_out_channels, 32),
            nn.ReLU(inplace=True),
        )

        # Standard classifier (no binary STE)
        self.classifier_bn = nn.BatchNorm1d(32)
        self.classifier = nn.Linear(32, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        h = self.reservoir(x)           # (B, T, 32)
        h = h.transpose(1, 2)           # (B, 32, T)
        dsconv_out = self.dsconv(h)     # (B, 32, T//2)
        fused = self.gate(h, dsconv_out)  # (B, 32, T//2)

        # Global Average Pooling
        pooled = fused.mean(dim=2)      # (B, 32)
        features = self.feature_dense(pooled)  # (B, 32)
        out = self.classifier(self.classifier_bn(features))  # (B, num_classes)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_kb(self):
        return self.count_parameters() * 4 / 1024

    def quantized_size_kb(self):
        return self.count_parameters() * 1 / 1024


# ---------------------------------------------------------------------------
# MSM Pre-trainer for Lite backbone
# ---------------------------------------------------------------------------
class MaskedSensorModelLite(nn.Module):
    """Masked Sensor Modeling pre-trainer for the Lite backbone."""

    def __init__(self, input_channels=6, reservoir_size=32, mask_ratio=0.15):
        super().__init__()
        self.input_channels = input_channels
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_channels))

        self.backbone_reservoir = EchoStateNetworkLite(
            input_channels, reservoir_size, learnable_sr=True
        )
        self.backbone_dsconv = DSConvEncoderLite(
            in_channels=reservoir_size, out_channels=32
        )

        # Reconstruct from dsconv features (32 channels)
        self.reconstruction_head = nn.Linear(32, input_channels)

    def forward(self, x, mask=None):
        batch_size, seq_len, channels = x.shape

        if mask is None:
            mask = _create_mask(batch_size, seq_len, self.mask_ratio, x.device)

        mask_expanded = mask.unsqueeze(-1).float()
        x_masked = x * (1.0 - mask_expanded) + self.mask_token * mask_expanded

        h = self.backbone_reservoir(x_masked)
        h = h.transpose(1, 2)
        dsconv_out = self.backbone_dsconv(h)  # (B, 32, T')

        # Upsample back to original temporal resolution
        recon_features = F.interpolate(
            dsconv_out, size=seq_len, mode='linear', align_corners=False
        )
        reconstruction = self.reconstruction_head(recon_features.transpose(1, 2))

        return reconstruction, mask


def _create_mask(batch_size, seq_len, mask_ratio, device):
    num_masked = max(1, int(seq_len * mask_ratio))
    mask = torch.zeros(batch_size, seq_len, device=device)
    for i in range(batch_size):
        indices = torch.randperm(seq_len, device=device)[:num_masked]
        mask[i, indices] = 1.0
    return mask


def msm_pretrain_lite(model, dataset, device, epochs=30, batch_size=128,
                      lr=0.0003, mask_ratio=0.15):
    """MSM pre-training loop for the Lite backbone."""
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
            bs, sl, ch = x.shape

            mask = _create_mask(bs, sl, mask_ratio, device)
            reconstruction, mask = model(x, mask=mask)

            mask_expanded = mask.unsqueeze(-1).float()
            masked_recon = reconstruction * mask_expanded
            masked_target = x * mask_expanded
            num_masked_elements = mask_expanded.sum().clamp(min=1.0)
            loss = ((masked_recon - masked_target) ** 2).sum() / (num_masked_elements * ch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"MSM Lite Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    return model


def transfer_masked_weights_lite(pretrained_msm, target_model):
    """Transfer MSM pre-trained weights to SensorFusionLite."""
    # Transfer reservoir weights (buffers + learnable SR)
    target_model.reservoir.load_state_dict(
        pretrained_msm.backbone_reservoir.state_dict(), strict=False
    )
    # Transfer DS-Conv weights
    target_model.dsconv.load_state_dict(
        pretrained_msm.backbone_dsconv.state_dict()
    )
    return target_model
