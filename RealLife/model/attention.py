import torch
import torch.nn as nn

__all__ = ["PatchMicroAttention"]


class PatchMicroAttention(nn.Module):

    def __init__(self, in_channels=48, seq_len=32, num_patches=8, d_model=32, num_heads=2, ff_dim=48):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = seq_len // num_patches
        self.patch_dim = in_channels * self.patch_size

        self.projection = nn.Linear(self.patch_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.xavier_uniform_(self.ffn[2].weight)

    def forward(self, x, return_attention=False):
        x = x.transpose(1, 2)
        batch = x.shape[0]
        x = x.reshape(batch, self.num_patches, self.patch_dim)
        x = self.projection(x) + self.pos_embedding

        x_norm = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(
                x_norm, x_norm, x_norm,
                need_weights=True, average_attn_weights=False
            )
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x = x + self.ffn(self.norm2(x))

        pooled = x.mean(dim=1)

        if return_attention:
            attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
            return pooled, attn_weights, attn_entropy
        return pooled
