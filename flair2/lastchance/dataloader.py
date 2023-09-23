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
        assert flag in ["train", "val", "test"]
        self.root = root
        self.isrunning = False
        self.flag = flag

        if flag == "train":
            self.trainpath = torch.load(root + "alltrainpaths.pth")

            tmp = sorted(self.trainpath.keys())
            tmp = [k for (i, k) in enumerate(tmp) if i % 4 == 0]
            self.valpath = {k: self.trainpath[k] for k in tmp}

            tmp = sorted(self.trainpath.keys())
            tmp = [k for (i, k) in enumerate(tmp) if i % 4 != 0]
            self.trainpath = {k: self.trainpath[k] for k in tmp}

            self.testpath = torch.load(root + "alltestpaths.pth")
            self.testpath.update(self.valpath)

            self.paths = {**self.testpath, **self.trainpath}
            self.trainpath = set(self.trainpath.keys())
            self.testpath = set(self.testpath.keys())

        if flag == "test":
            self.paths = torch.load(root + "alltestpaths.pth")

        if flag == "val":
            self.paths = torch.load(root + "alltrainpaths.pth")
            tmp = sorted(self.paths.keys())
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
        sen = torch.Tensor(sentinel[:, :, row : row + 40, col : col + 40])
        sen = torch.nan_to_num(sen)
        sen = torch.clamp(sen, -2, 2)

        if self.flag == "test":
            return torch.Tensor(x), sen
        if self.flag == "val" or k in self.trainpath:
            with rasterio.open(self.root + self.paths[k]["label"]) as src:
                y = torch.Tensor(numpy.clip(src.read(1), 1, 13) - 1)
            return torch.Tensor(x), sen, y
        else:
            tmp = torch.zeros(512, 512)
            tmp[0][0] = -1
            return torch.Tensor(x), sen, tmp

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize=16):
        x = torch.zeros(batchsize, 5, 512, 512)
        sen = torch.zeros(batchsize, 10, 32, 40, 40)
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


class MyNet6(torch.nn.Module):
    def __init__(self):
        super(MyNet6, self).__init__()
        tmp = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
        del tmp[7]
        del tmp[6]
        del tmp[5]
        with torch.no_grad():
            old = tmp[0][0].weight / 2
            tmp[0][0] = torch.nn.Conv2d(
                6, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            tmp[0][0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))

            old = tmp[3][0].block[0][0].weight
            tmp[3][0].block[0][0] = torch.nn.Conv2d(48, 192, kernel_size=3, padding=1, bias=False)
            tmp[3][0].block[0][0].weight= torch.nn.Parameter(old)

            old = tmp[4][0].block[1][0].weight
            tmp[4][0].block[1][0] = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False)
            tmp[4][0].block[1][0].weight= torch.nn.Parameter(old)
            
        self.classiflow = torch.nn.Conv2d(128, 13, kernel_size=1)

    def forward(self, x,s):
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.5
        x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        x = self.backbone(x)
        plow = self.classiflow(x)
        plow = torch.nn.functional.interpolate(plow, size=(512, 512), mode="bilinear")

        xs = torch.cat([x, s], dim=1)
        xs = self.lrelu(self.merge1(xs))
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge2(xs))
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge3(xs))

        f1 = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        f2 = torch.nn.functional.interpolate(xs, size=(128, 128), mode="bilinear")
        f = torch.cat([f1, f2, hr], dim=1)
        f2 = self.lrelu(self.decod1(f))
        f = torch.cat([f1, f2, hr], dim=1)
        f2 = self.lrelu(self.decod2(f))
        f = torch.cat([f1, f2, hr], dim=1)
        f2 = self.lrelu(self.decod3(f))
        f = torch.cat([f1, f2, hr], dim=1)
        f2 = self.lrelu(self.decod4(f))
        f = torch.cat([f1, f2, hr], dim=1)
        p = self.classif(f)

        return p

if __name__ == "__main__":
    import os

    os.system("rm -rf build")
    os.system("mkdir build")

    os.system("/d/achanhon/miniconda3/bin/python -u train.py")
    quit()
