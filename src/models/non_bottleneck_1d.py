import torch
import torch.nn as nn
import torch.nn.functional as F


class NonBottleneck1DResidual(nn.Module):
    def __init__(self, channels, dropout_probability, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.batch_norm_1 = nn.BatchNorm2d(channels, eps=1e-3)

        self.conv3x1_2 = nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(channels, channels, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.batch_norm_2 = nn.BatchNorm2d(channels, eps=1e-3)

        self.dropout = nn.Dropout2d(dropout_probability)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.batch_norm_1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.batch_norm_2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + input)
