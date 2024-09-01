import torch
import torch.nn as nn

from .non_bottleneck_1d import NonBottleneck1DResidual


class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        return output


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(Upsampler(128, 64))

        for _ in range(2):
            self.layers.append(NonBottleneck1DResidual(64, 0, 1))

        self.layers.append(Upsampler(64, 16))

        for _ in range(2):
            self.layers.append(NonBottleneck1DResidual(16, 0, 1))

        self.layers.append(nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True))

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        return output
