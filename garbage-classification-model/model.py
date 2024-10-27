import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.resample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor):
        identity: torch.Tensor = self.resample(x)
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class MinResNet(nn.Module):
    def __init__(self):
        super().__init__()
        dim_hiddens = [8, 16, 32, 64, 128, 256]
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(
            3, dim_hiddens[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim_hiddens[0])
        self.layer1 = ResidualBlock(dim_hiddens[0], dim_hiddens[1])
        self.layer2 = ResidualBlock(dim_hiddens[1], dim_hiddens[2])
        self.layer3 = ResidualBlock(dim_hiddens[2], dim_hiddens[3])
        self.layer4 = ResidualBlock(dim_hiddens[3], dim_hiddens[4])
        self.layer5 = ResidualBlock(dim_hiddens[4], dim_hiddens[5])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim_hiddens[5], 12)

    def forward(self, x: torch.Tensor):
        # (bs, 3, 256, 256)
        x = self.conv1(x)  # (bs, 8, 128, 128)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # (bs, 8, 64, 64)
        x = self.layer1(x)  # (bs, 16, 64, 64)
        x = self.pool(x)  # (bs, 16, 32, 32)
        x = self.layer2(x)  # (bs, 32, 32, 32)
        x = self.pool(x)  # (bs, 32, 16, 16)
        x = self.layer3(x)  # (bs, 64, 16, 16)
        x = self.pool(x)  # (bs, 64, 8, 8)
        x = self.layer4(x)  # (bs, 128, 8, 8)
        x = self.pool(x)  # (bs, 128, 4, 4)
        x = self.layer5(x)  # (bs, 256, 4, 4)
        x = self.avgpool(x)  # (bs, 256, 1, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
