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
    cmt = torch.transpose(cm, 0, 1)

    accu = 0
    for i in range(12):
        accu += cm[i][i]
    accu /= cm[0:12, 0:12].flatten().sum()

    iou = 0
    for i in range(12):
        inter = cm[i][i]
        union = cm[i].sum() + cmt[i].sum() - cm[i][i] + (cm[i][i] == 0)
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
            x = numpy.stack([r, g, b, i, e], axis=0)

        sentinel = readSEN(self.root + self.paths[k]["sen"])
        row, col = self.paths[k]["coord"]
        sen = torch.Tensor(sentinel[:, row : row + 40, col : col + 40])
        sen = torch.nan_to_num(sen)
        sen = torch.clamp(sen, 0, 1)

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


class Baseline(torch.nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
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

        self.decod1 = torch.nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.decod2 = torch.nn.Conv2d(224, 128, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(288, 256, kernel_size=1)
        self.decod4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.5
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[2](self.backbone[1](self.backbone[0](x)))
        x = self.backbone[5](self.backbone[4](self.backbone[3](hr)))
        plow = self.classiflow(x)
        plow = torch.nn.functional.interpolate(plow, size=(512, 512), mode="bilinear")

        x = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        hr = torch.nn.functional.gelu(self.decod1(hr))
        hr = torch.cat([hr, x], dim=1)
        hr = torch.nn.functional.gelu(self.decod2(hr))
        hr = torch.cat([hr, x], dim=1)
        hr = torch.nn.functional.gelu(self.decod3(hr))
        hr = torch.nn.functional.gelu(self.decod4(hr))
        p = self.classif(hr)
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")

        return p + 0.2 * plow, hr


class MyNet(torch.nn.Module):
    def __init__(self, baseline):
        super(MyNet, self).__init__()
        self.baseline = torch.load(baseline)

        self.conv1 = torch.nn.Conv2d(200, 512, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(512, 512, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(512, 100, kernel_size=1)

        self.merge1 = torch.nn.Conv2d(356, 100, kernel_size=1)
        self.merge2 = torch.nn.Conv2d(356, 100, kernel_size=1)

        self.merge31 = torch.nn.Conv2d(356, 256, kernel_size=1)
        self.merge32 = torch.nn.Conv2d(356, 100, kernel_size=3, padding=1)

        self.merge4 = torch.nn.Conv2d(356, 448, kernel_size=3)
        self.merge5 = torch.nn.Conv2d(448, 512, kernel_size=1)
        self.classif = torch.nn.Conv2d(512, 13, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x, s):
        with torch.no_grad():
            px, hr = self.baseline(x)

        s = self.lrelu(self.conv1(s))
        s = self.lrelu(self.conv2(s))
        s = self.lrelu(self.conv3(s))
        s = torch.nn.functional.interpolate(s, size=(128, 128), mode="bilinear")

        xs = torch.cat([hr, s], dim=1)
        xs = self.lrelu(self.merge1(xs))
        xs = torch.cat([hr, xs], dim=1)
        xs = self.lrelu(self.merge2(xs))
        xs = torch.cat([hr, xs], dim=1)

        s = torch.nn.functional.relu(self.merge31(xs))
        s = (s - 1) / 1000 * (s > 1).float() + s * (s <= 1).float()
        xs = torch.nn.functional.gelu(self.merge32(xs))
        xs = torch.cat([hr * s, xs], dim=1)

        xs = torch.nn.functional.gelu(self.merge4(xs))
        xs = torch.nn.functional.gelu(self.merge5(xs))

        p = self.classif(xs)
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")

        return p + px


if __name__ == "__main__":
    import os

    os.system("rm -rf build")
    os.system("mkdir build")

    os.system("/d/achanhon/miniconda3/bin/python -u loaderBaseline.py")
    os.system("/d/achanhon/miniconda3/bin/python -u train.py")
    os.system("/d/achanhon/miniconda3/bin/python -u val.py")
    os.system("/d/achanhon/miniconda3/bin/python -u test.py")
    quit()
