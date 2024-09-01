import torch.nn as nn

from . import Encoder, Decoder
from .encoder import EncoderLowDilation


class ERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(encoder or Encoder(num_classes))
        self.layers.append(Decoder(num_classes))

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        return output


class ERFNetLowDilation(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(encoder or EncoderLowDilation(num_classes))
        self.layers.append(Decoder(num_classes))

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        return output
