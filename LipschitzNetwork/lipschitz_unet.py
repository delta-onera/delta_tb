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
    def __init__(self):
        super(MinMax, self).__init__()

    def forward(self, inputs):
        assert len(inputs.shape) == 4  # BxCxWxH
        assert inputs.shape[1] % 2 == 0  # C%2==0

        tmp = torch.transpose(inputs, 1, 2)  # BxWxCxH
        tmpmax = F.max_pool2d(tmp, kernel_size=(2, 1), stride=(2, 1))  # BxWxC/2xH
        tmpmin = -F.max_pool2d(-tmp, kernel_size=(2, 1), stride=(2, 1))  # BxWxC/2xH

        tmp = torch.cat([tmpmin, tmpmax], dim=2)  # BxWxCxH
        return torch.transpose(tmp, 2, 1)  # BxCxWxH


class UNET(nn.Module):
    def __init__(self, nbclasses=2, nbchannel=3, pretrained=""):
        super(UNET, self).__init__()

        self.nbclasses = nbclasses
        self.nbchannel = nbchannel
        self.minmax = MinMax()

        self.conv1 = nn.Conv2d(self.nbchannel, 32, kernel_size=9, padding=4)
        self.pool1 = nn.Conv2d(self.nbchannel, 64, kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.Conv2d(self.nbchannel, 128, kernel_size=4, stride=4, padding=0)

        self.l1 = nn.Conv2d(224, 128, kernel_size=1, padding=0)
        self.l2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.l3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)

        self.encoding1 = nn.Conv2d(128, 32, kernel_size=1, padding=0)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.pool3 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.Conv2d(128, 256, kernel_size=4, stride=4, padding=0)

        self.l4 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.l5 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.l6 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.l7 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.encoding2 = nn.Conv2d(512, 32, kernel_size=1, padding=0)

        self.d1 = nn.Conv2d(96, 128, kernel_size=1, padding=0)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.d4 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.decoding = nn.Conv2d(128, self.nbclasses, kernel_size=1, padding=0)

    def forward(self, x):
        code1 = self.minmax(self.conv1(x))

        x1 = F.max_pool2d(code1, kernel_size=4, stride=4)
        x2 = self.minmax(self.pool1(F.avg_pool2d(x, kernel_size=2, stride=2)))
        x4 = self.minmax(self.pool2(x))

        x4 = torch.cat([x1 / 3, x2 / 3, x4 / 3], dim=1)
        x4 = self.minmax(self.l1(x4))
        x4 = self.minmax(self.l2(x4))
        x4 = self.minmax(self.l3(x4))

        code4 = self.encoding1(x4)

        x16 = self.minmax(self.pool4(x4))
        x8 = self.minmax(self.pool3(F.max_pool2d(x4, kernel_size=2, stride=2)))
        x4 = F.max_pool2d(self.minmax(self.conv2(x4)), kernel_size=4, stride=4)

        x16 = torch.cat([x4 / 3, x8 / 3, x16 / 3], dim=1)
        x16 = self.minmax(self.l4(x16))
        x16 = self.minmax(self.l5(x16))
        x16 = self.minmax(self.l6(x16))
        x16 = self.minmax(self.l7(x16))

        code16 = self.encoding2(x16)

        code16 = F.interpolate(code16, size=x.shape[2:4], mode="nearest")
        code4 = F.interpolate(code4, size=x.shape[2:4], mode="nearest")
        x = torch.cat([code1 / 3, code4 / 16 / 3, code16 / 256 / 3], dim=1)

        x = self.minmax(self.d1(x))
        x = self.minmax(self.d2(x))
        x = self.minmax(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = self.minmax(self.d4(x))
        x = self.decoding(x)
        return x

    def normalize(self, force):
        with torch.no_grad():
            for layer in self._modules:
                if layer in ["minmax"]:
                    continue

                norms = self._modules[layer].weight[:].norm(2).clamp_min(0.0001)
                if layer in ["conv1", "conv2", "conv3"]:
                    W = self._modules[layer].weight.shape[-1]
                    H = self._modules[layer].weight.shape[-2]
                    norm = torch.sqrt(torch.sum(norms * norms) * W * H)
                else:
                    norm = torch.sqrt(torch.sum(norms * norms))

                if norm > 3 or force:
                    self._modules[layer].weight *= 1.0 / norm
                    self._modules[layer].bias *= 1.0 / norm


if __name__ == "__main__":
    if True:
        A = torch.rand((3, 5))
        x = torch.rand(5) - 0.5
        for i in range(5000):
            with torch.no_grad():
                x = x / x.norm(2)

                norms = A[:].norm(2)
                norm = torch.sqrt(torch.sum(norms * norms))
                A = A / norm

            if A.grad is not None:
                A.grad.zeros_()
            if x.grad is not None:
                x.grad.zeros_()
            A.requires_grad_(True)
            x.requires_grad_(True)
            loss = torch.norm(torch.mv(A, x))
            if i % 50 == 0:
                print(x, loss)
            loss.backward()
            A = A + 0.001 * A.grad
            x = x + 0.001 * x.grad

    else:

        net = UNET()

        print("before normalization")
        print(net.conv1.weight[1][1])

        net.normalize()

        print("after normalization")
        print(net.conv1.weight[1][1])

        tmp = torch.rand(4, 3, 128, 128)
        print(net(tmp).shape)
