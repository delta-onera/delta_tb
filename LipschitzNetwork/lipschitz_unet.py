import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def loadpretrained(model, correspondance, path):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()

    for name1, name2 in correspondance:
        fw, fb = False, False
        for name, param in pretrained_dict.items():
            if name == name1 + ".weight":
                model_dict[name2 + ".weight"].copy_(param)
                fw = True
            if name == name1 + ".bias":
                model_dict[name2 + ".bias"].copy_(param)
                fb = True
        if (not fw) or (not fb):
            print(name2 + " not found")
            quit()
    model.load_state_dict(model_dict)


class MinMax(nn.Module):
    def __init__(self, debug):
        super(MinMax, self).__init__()
        self.debug = debug

    def forward(self, inputs):
        if self.debug:
            return F.leaky_relu(inputs)

        assert len(inputs.shape) == 4  # BxCxWxH
        assert inputs.shape[1] % 2 == 0  # C%2==0

        tmp = torch.transpose(inputs, 1, 2)  # BxWxCxH
        tmpmax = F.max_pool2d(tmp, kernel_size=(2, 1), stride=(2, 1))  # BxWxC/2xH
        tmpmin = -F.max_pool2d(-tmp, kernel_size=(2, 1), stride=(2, 1))  # BxWxC/2xH

        tmp = torch.cat([tmpmin, tmpmax], dim=2)  # BxWxCxH
        return torch.transpose(tmp, 2, 1)  # BxCxWxH


class AbsActivation(nn.Module):
    def __init__(self, debug):
        super(AbsActivation, self).__init__()
        self.debug = debug

    def forward(self, inputs):
        if self.debug:
            return F.leaky_relu(inputs)
        else:
            return torch.abs(inputs)


class UNET(nn.Module):
    def __init__(self, nbclasses=2, nbchannel=3, pretrained="", debug=False):
        super(UNET, self).__init__()

        self.nbclasses = nbclasses
        self.nbchannel = nbchannel
        self.minmax = MinMax(debug)
        self.absactivation = AbsActivation(debug)

        self.conv11 = nn.Conv2d(self.nbchannel, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.gradientdoor = nn.Conv2d(512, 32, kernel_size=1)

        self.conv43d = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.conv33d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.conv22d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.final1 = nn.Conv2d(160, 64, kernel_size=3, padding=1)
        self.final2 = nn.Conv2d(64, self.nbclasses, kernel_size=1)

        if pretrained != "":
            correspondance = []
            if self.nbchannel == 3:
                correspondance.append(("features.0", "conv11"))
            correspondance.append(("features.2", "conv12"))
            correspondance.append(("features.5", "conv21"))
            correspondance.append(("features.7", "conv22"))
            correspondance.append(("features.10", "conv31"))
            correspondance.append(("features.12", "conv32"))
            correspondance.append(("features.14", "conv33"))
            correspondance.append(("features.17", "conv41"))
            correspondance.append(("features.19", "conv42"))
            correspondance.append(("features.21", "conv43"))
            correspondance.append(("features.24", "conv51"))
            correspondance.append(("features.26", "conv52"))
            correspondance.append(("features.28", "conv53"))
            loadpretrained(self, correspondance, pretrained)

    def forward(self, x):
        x = F.leaky_relu(self.conv11(x)) * 2
        x1 = F.leaky_relu(self.conv12(x)) * 2

        x = F.max_pool2d(x1, kernel_size=2, stride=2)
        x = self.minmax(self.conv21(x))
        x2 = self.minmax(self.conv22(x))

        x = F.max_pool2d(x2, kernel_size=2, stride=2)
        x = self.minmax(self.conv31(x))
        x = self.absactivation(self.conv32(x))
        x3 = self.minmax(self.conv33(x))

        x = F.max_pool2d(x3, kernel_size=2, stride=2)
        x = self.minmax(self.conv41(x))
        x = self.absactivation(self.conv42(x))
        x4 = self.minmax(self.conv43(x))

        x = F.max_pool2d(x4, kernel_size=2, stride=2)
        x = self.minmax(self.conv51(x))
        x = self.absactivation(self.conv52(x))
        x5 = self.minmax(self.conv53(x))

        x = self.gradientdoor(x5)
        x = F.interpolate(x, size=x1.shape[2:4], mode="nearest")

        x5 = F.interpolate(x5, size=x4.shape[2:4], mode="nearest")
        x4 = torch.cat((x5, x4), 1)
        x4 = self.minmax(self.conv43d(x4))
        x4 = self.absactivation(self.conv42d(x4))
        x4 = self.minmax(self.conv41d(x4))

        x4 = F.interpolate(x4, size=x3.shape[2:4], mode="nearest")
        x3 = torch.cat((x4, x3), 1)
        x3 = self.minmax(self.conv33d(x3))
        x3 = self.absactivation(self.conv32d(x3))
        x3 = self.minmax(self.conv31d(x3))

        x3 = F.interpolate(x3, size=x2.shape[2:4], mode="nearest")
        x2 = torch.cat((x3, x2), 1)
        x2 = self.minmax(self.conv22d(x2))
        x2 = self.absactivation(self.conv21d(x2))

        x2 = F.interpolate(x2, size=x1.shape[2:4], mode="nearest")
        x1 = torch.cat((x2, x1, x), 1)

        x = F.leaky_relu(self.final1(x1)) * 2
        x = self.final2(x)
        return x

    def normalize(self):
        with torch.no_grad():
            for layer in self._modules:
                if layer in ["minmax", "absactivation"]:
                    continue
                denom = self._modules[layer].weight.norm(2).clamp_min(0.00000001)
                self._modules[layer].weight *= 1.0 / denom
                self._modules[layer].bias *= 1.0 / denom


if __name__ == "__main__":
    net = UNET()

    print("before normalization")
    print(net.conv22.weight[1][1][0:10, 0:10])
    print(net.conv41d.weight[1][1][0:10, 0:10])
    print(net.final1.weight[1][1][0:10, 0:10])

    net.normalize()

    print("after normalization")
    print(net.conv22.weight[1][1][0:10, 0:10])
    print(net.conv41d.weight[1][1][0:10, 0:10])
    print(net.final1.weight[1][1][0:10, 0:10])

    tmp = torch.rand(4, 3, 128, 128)
    print(net(tmp).shape)
