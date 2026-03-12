import torch.nn as nn

__all__ = ["DSConvEncoder"]


class DepthwiseSeparableBlock(nn.Module):

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


class DSConvEncoder(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(in_channels, 128, kernel_size=5, stride=1, padding=2),
            DepthwiseSeparableBlock(128, 128, kernel_size=5, stride=2, padding=2),
            DepthwiseSeparableBlock(128, 128, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.blocks(x)
