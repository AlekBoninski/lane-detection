import torch
import torch.nn as nn
import torch.nn.functional as F

from .non_bottleneck_1d import NonBottleneck1DResidual


class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class Encoder(nn.Module):
    def __init__(self, num_classes, predict=False):
        super().__init__()

        self.predict = predict

        self.layers = nn.ModuleList()
        self.layers.append(Downsampler(3, 16))
        self.layers.append(Downsampler(16, 64))

        for _ in range(5):
            self.layers.append(NonBottleneck1DResidual(64, 0.03, 1))

        self.layers.append(Downsampler(64, 128))

        for _ in range(2):
            self.layers.append(NonBottleneck1DResidual(128, 0.3, 2))
            self.layers.append(NonBottleneck1DResidual(128, 0.3, 4))
            self.layers.append(NonBottleneck1DResidual(128, 0.3, 8))
            self.layers.append(NonBottleneck1DResidual(128, 0.3, 16))

        # Only for standalone encoder
        self.encoder_out = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        if self.predict:
            output = self.encoder_out(output)

        return output


class EncoderLowDilation(nn.Module):
    def __init__(self, num_classes, predict=False):
        super().__init__()

        self.predict = predict

        self.layers = nn.ModuleList()
        self.layers.append(Downsampler(3, 16))
        self.layers.append(Downsampler(16, 64))

        for _ in range(5):
            self.layers.append(NonBottleneck1DResidual(64, 0.03, 1))

        self.layers.append(Downsampler(64, 128))

        for _ in range(2):
            self.layers.append(NonBottleneck1DResidual(128, 0.3, 1))
            self.layers.append(NonBottleneck1DResidual(128, 0.3, 2))

        # Only for standalone encoder
        self.encoder_out = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        if self.predict:
            output = self.encoder_out(output)

        return output
