import torch
import torch.nn as nn


class MinMax(nn.Module):
    def __init__(self):
        super(MinMax, self).__init__()

    def forward(self, inputs):
        assert len(inputs.shape) == 4  # BxCxWxH
        assert inputs.shape[1] % 2 == 0  # C%2==0

        tmp = torch.transpose(inputs, 1, 2)  # BxWxCxH
        tmpmax = nn.functional.max_pool2d(
            tmp, kernel_size=(2, 1), stride=(2, 1)
        )  # BxWxC/2xH
        tmpmin = -nn.functional.max_pool2d(
            -tmp, kernel_size=(2, 1), stride=(2, 1)
        )  # BxWxC/2xH

        tmp = torch.cat([tmpmin, tmpmax], dim=2)  # BxWxCxH
        return torch.transpose(tmp, 2, 1)  # BxCxWxH


class UNET(nn.Module):
    def __init__(self, nbclasses=2, nbchannel=3, debug=False):
        super(UNET, self).__init__()

        self.nbclasses = nbclasses
        self.nbchannel = nbchannel

        if debug:
            self.minmax = nn.LeakyReLU()
        else:
            self.minmax = MinMax()

        self.conv1 = nn.Conv2d(self.nbchannel, 32, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(self.nbchannel, 32, kernel_size=9, padding=4)
        self.conv4 = nn.Conv2d(self.nbchannel * 2, 32, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(self.nbchannel * 2, 32, kernel_size=5, padding=2)

        self.l1 = nn.Conv2d(32, 32, kernel_size=1)
        self.l2 = nn.Conv2d(64, 64, kernel_size=1)
        self.l4 = nn.Conv2d(96, 96, kernel_size=1)

        self.e1 = nn.Conv2d(128, 256, kernel_size=1)
        self.e2 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.e3 = nn.Conv2d(512, 1024, kernel_size=1)
        self.e31 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.e32 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.e4 = nn.Conv2d(1024, 128, kernel_size=1)

        self.d1 = nn.Conv2d(32 + 64 + 96 + 128, 256, kernel_size=1)
        self.d2 = nn.Conv2d(256, 256, kernel_size=1)
        self.d3 = nn.Conv2d(256, self.nbclasses, kernel_size=1)

        self.layers = [
            layer for layer in self._modules if layer not in ["minmax", "d3"]
        ]

    def forward(self, x):
        x1 = self.minmax(self.conv1(x) / 81)
        x1 = self.minmax(self.l1(x1))

        x2 = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x2 = self.minmax(self.conv2(x2) / 81)
        x2 = torch.cat(
            [nn.functional.max_pool2d(x1, kernel_size=2, stride=2) / 2, x2 / 2], dim=1
        )
        x2 = self.minmax(self.l2(x2))

        x4_p = nn.functional.max_pool2d(x, kernel_size=4, stride=4)
        x4_m = -nn.functional.max_pool2d(-x, kernel_size=4, stride=4)
        x4 = torch.cat([x4_p / 2, x4_m / 2], dim=1)
        x4 = self.minmax(self.conv4(x4) / 25)
        x4 = torch.cat(
            [nn.functional.max_pool2d(x2, kernel_size=2, stride=2) / 2, x4 / 2], dim=1
        )
        x4 = self.minmax(self.l4(x4))

        x8_p = nn.functional.max_pool2d(x, kernel_size=8, stride=8)
        x8_m = -nn.functional.max_pool2d(-x, kernel_size=8, stride=8)
        x8 = torch.cat([x8_p / 2, x8_m / 2], dim=1)
        x8 = self.minmax(self.conv8(x8) / 25)
        x8 = torch.cat(
            [nn.functional.max_pool2d(x4, kernel_size=2, stride=2) / 2, x8 / 2], dim=1
        )

        x8 = self.minmax(self.e1(x8))
        x8 = self.minmax(self.e2(x8) / 25)
        x8 = self.minmax(self.e3(x8))
        x8 = self.minmax(self.e31(x8))
        x8 = self.minmax(self.e32(x8))
        x8 = self.minmax(self.e4(x8))

        x8 = nn.functional.interpolate(x8, size=x.shape[2:4], mode="nearest")
        x4 = nn.functional.interpolate(x4, size=x.shape[2:4], mode="nearest")
        x2 = nn.functional.interpolate(x2, size=x.shape[2:4], mode="nearest")

        x = torch.cat([x1 / 4, x2 / 2 / 2 / 4, x4 / 4 / 4 / 4, x8 / 8 / 8 / 4], dim=1)

        x = self.minmax(self.d1(x))
        x = self.minmax(self.d2(x))
        x = self.minmax(self.d3(x))
        return x

    def getNorm(self, tensor):
        return torch.sqrt(tensor.norm(2).clamp_min(0.0001))

    def getLipschitzbound(self):
        K = self.getNorm(self._modules["d3"].weight[:])
        for layer in self.layers:
            K *= self.getNorm(self._modules[layer].weight[:])
        return K
