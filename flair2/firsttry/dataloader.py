import torch
import numpy
import queue
import threading
import rasterio
import random
from functools import lru_cache


@lru_cache
def readSEN(path):
    return numpy.load(path)


class FLAIR2(threading.Thread):
    def __init__(self, flag="test", root="/scratchf/CHALLENGE_IGN/FLAIR_2/"):
        threading.Thread.__init__(self)
        assert flag in ["train", "val", "trainval", "test"]
        self.root = root
        self.isrunning = False
        self.flag = flag
        if flag == "test":
            self.paths = torch.load(root + "alltestpaths.pth")
        else:
            self.paths = torch.load(root + "alltrainpaths.pth")

        tmp = sorted(self.paths.keys())
        if flag == "train":
            tmp = [k for (i, k) in enumerate(tmp) if i % 3 != 0]
        if flag == "val":
            tmp = [k for (i, k) in enumerate(tmp) if i % 3 == 0]
        self.paths = {k: self.paths[k] for k in tmp}

    def get(self, k):
        assert k in self.paths
        with rasterio.open(self.root + self.paths[k]["image"]) as src:
            r = numpy.clip(src.read(1), 0, 255)
            g = numpy.clip(src.read(2), 0, 255)
            b = numpy.clip(src.read(3), 0, 255)
            i = numpy.clip(src.read(4), 0, 255)
            e = numpy.clip(src.read(5), 0, 255)
            x = numpy.stack([r, g, b, i, e], axis=0) * 255

        sentinel = readSEN(self.root + self.paths[k]["sen"])
        assert sentinel.shape[0:2] == (10, 20)
        row, col = self.paths[k]["coord"]
        sen = sentinel[:, :, row : row + 40, col : col + 40]
        sen = torch.Tensor(sen).flatten(0, 1)

        if self.flag != "test":
            with rasterio.open(self.root + self.paths[k]["label"]) as src:
                y = torch.Tensor(numpy.clip(src.read(1), 0, 13))
            return torch.Tensor(x), sen, y
        else:
            return torch.Tensor(x), sen

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize=16):
        x = torch.zeros(batchsize, 5, 512, 512)
        sen = torch.zeros(batchsize, 200, 40, 40)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], sen[i], y[i] = self.getCrop()
        return x, sen, y

    def run(self):
        assert self.isrunning == False
        self.isrunning = True
        self.q = queue.Queue(maxsize=100)

        while True:
            I = list(self.paths.keys())
            random.shuffle(I)
            for i in I:
                self.q.put(self.get(i), block=True)


import torchvision


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        tmp = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
        with torch.no_grad():
            old = tmp[0][0].weight / 2
            tmp[0][0] = torch.nn.Conv2d(
                6, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            tmp[0][0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))
        del tmp[7]
        del tmp[6]
        self.backbone = tmp
        self.classiflow = torch.nn.Conv2d(160, 13, kernel_size=1)

        self.conv1 = torch.nn.Conv2d(200, 200, kernel_size=3, groups=20)
        self.conv2 = torch.nn.Conv2d(200, 200, kernel_size=3, groups=20)
        self.conv3 = torch.nn.Conv2d(200, 200, kernel_size=3, groups=20)
        self.conv4 = torch.nn.Conv2d(200, 200, kernel_size=3, groups=20)
        self.conv5 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv6 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv7 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv8 = torch.nn.Conv2d(200, 160, kernel_size=1)

        self.merge1 = torch.nn.Conv2d(320, 160, kernel_size=1, groups=8)
        self.merge2 = torch.nn.Conv2d(320, 160, kernel_size=1, groups=8)
        self.merge3 = torch.nn.Conv2d(320, 160, kernel_size=1)
        self.merge4 = torch.nn.Conv2d(320, 160, kernel_size=1)
        self.classif = torch.nn.Conv2d(320, 13, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x, s, keepEFF=False):
        x = ((x / 255) - 0.5) / 0.5
        xm = torch.zeros(x.shape[0], 1, 512, 512)
        x = torch.cat([x, xm], dim=1)
        if keepEFF:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        plow = self.classiflow(x)

        s = self.lrelu(self.conv1(s))
        s = self.lrelu(self.conv2(s))
        s = self.lrelu(self.conv3(s))
        s = self.lrelu(self.conv4(s))
        s = self.lrelu(self.conv5(s))
        s = self.lrelu(self.conv6(s))
        s = self.lrelu(self.conv7(s))
        s = self.lrelu(self.conv8(s))

        s = torch.cat([s, x], dim=1)
        s = torch.nn.functional.gelu(self.merge1(s))
        s = torch.cat([s, x], dim=1)
        s = torch.nn.functional.gelu(self.merge2(s))
        s = torch.cat([s, x], dim=1)
        s = torch.nn.functional.gelu(self.merge3(s))
        s = torch.cat([s, x], dim=1)
        s = torch.nn.functional.gelu(self.merge4(s))
        s = torch.cat([s, x], dim=1)

        p = self.classif(s)
        p = p + plow * 0.5
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
        return p


if __name__ == "__main__":
    net = MyNet()
    print(net(torch.rand(2, 5, 512, 512), torch.rand(2, 200, 40, 40)).shape)
    quit()

    import os

    data = FLAIR2("train")
    data.start()
    x, s, y = data.getBatch()
    y = y[0] / 13
    x = x[0, 3, :, :] / 255
    s = s[0, 3, :, :] / 4
    torchvision.utils.save_image(x, "build/x.png")
    torchvision.utils.save_image(y, "build/y.png")
    torchvision.utils.save_image(s, "build/s.png")
    os._exit(0)
