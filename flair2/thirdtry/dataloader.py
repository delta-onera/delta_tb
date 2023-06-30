import torch
import numpy
import queue
import threading
import rasterio
import random
from functools import lru_cache


def confusion(y, z):
    cm = torch.zeros(13, 13).cuda()
    for a in range(13):
        for b in range(13):
            cm[a][b] = ((z == a).float() * (y == b).float()).sum()
    return cm


def perf(cm):
    accu = 0
    for i in range(12):
        accu += cm[i][i]
    accu /= cm[0:12, 0:12].flatten().sum()

    iou = 0
    for i in range(12):
        inter = cm[i][i]
        union = cm[i].sum() + (cm[:, i]).sum() - cm[i][i] + (cm[i][i] == 0)
        iou += inter / union

    return (iou / 12 * 100, accu * 100)


@lru_cache
def readSEN(path):
    return numpy.load(path)


class FLAIR2(threading.Thread):
    def __init__(self, flag="test", root="/d/achanhon/FLAIR_2/"):
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
            tmp = [k for (i, k) in enumerate(tmp) if i % 4 != 0]
        if flag == "val":
            tmp = [k for (i, k) in enumerate(tmp) if i % 4 == 0]
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
                y = torch.Tensor(numpy.clip(src.read(1), 1, 13) - 1)
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


class SentinelNet(torch.nn.Module):
    def __init__(self):
        super(SentinelNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(200, 160, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(160, 160, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(160, 160, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(160, 160, kernel_size=1)
        self.conv5 = torch.nn.Conv2d(160, 160, kernel_size=1)

        self.merge1 = torch.nn.Conv2d(320, 160, kernel_size=1)
        self.merge2 = torch.nn.Conv2d(320, 160, kernel_size=1)
        self.merge3 = torch.nn.Conv2d(320, 160, kernel_size=1)
        self.merge4 = torch.nn.Conv2d(320, 160, kernel_size=1)
        self.classif = torch.nn.Conv2d(320, 2, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x, s):
        s = self.lrelu(self.conv1(s))
        s = self.lrelu(self.conv2(s))
        s = self.lrelu(self.conv3(s))
        s = self.lrelu(self.conv4(s))
        s = self.lrelu(self.conv5(s))

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
        return torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")


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

        self.spatial = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large()
        with torch.no_grad():
            old = self.spatial.backbone["0"][0].weight / 2
            self.spatial.backbone["0"][0] = torch.nn.Conv2d(
                6, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            self.spatial.backbone["0"][0].weight = torch.nn.Parameter(
                torch.cat([old, old], dim=1)
            )
        self.spatial.classifier[4] = torch.nn.Identity()
        self.classifierhigh = torch.nn.Conv2d(256, 13, kernel_size=(1, 1))

        heads = torch.nn.ModuleDict()
        for i in range(12):
            heads[i] = SentinelNet()
        self.w = torch.ones(13).cuda()

    def forward(self, x, s, nohead=False):
        x = ((x / 255) - 0.5) / 0.5
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        x = torch.cat([x, xm], dim=1)
        if nohead:
            hr = self.spatial(2 * x)["out"]
            phigh = self.classifierhigh(hr)
            x = self.backbone(x)
            plow = self.classiflow(x)
            plow = torch.nn.functional.interpolate(
                plow, size=(512, 512), mode="bilinear"
            )
            return plow + phigh

        # else
        with torch.no_grad():
            hr = self.spatial(2 * x)["out"]
            phigh = self.classifierhigh(hr)
            x = self.backbone(x)
            plow = self.classiflow(x)
            plow = torch.nn.functional.interpolate(
                plow, size=(512, 512), mode="bilinear"
            )

        P = {}
        for i in range(12):
            P[i] = self.heads[i](x, s)
        P["low"] = plow
        P["high"] = phigh
        return P

    def merge(self, P):
        p = torch.nn.functional.softmax(P["low"] + P["high"], dim=1) * self.w[-1]
        for i in range(12):
            P[i] = torch.nn.functional.softmax(P[i], dim=1)
            p[i] = p[i] + P[i][1] * self.w[i]
        return p


if __name__ == "__main__":
    import os

    os.system("rm -rf build")
    os.system("mkdir build")
    os.system("/d/achanhon/miniconda3/bin/python -u train.py")
    os.system("/d/achanhon/miniconda3/bin/python -u val.py")
    os.system("/d/achanhon/miniconda3/bin/python -u test.py")
    quit()
