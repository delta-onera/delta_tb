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


class MyNet5(torch.nn.Module):
    def __init__(self):
        super(MyNet5, self).__init__()
        tmp = torchvision.models.convnext_small(weights="DEFAULT").features
        with torch.no_grad():
            old = tmp[0][0].weight / 2
            oldb = tmp[0][0].bias
            tmp[0][0] = torch.nn.Conv2d(6, 96, kernel_size=(4, 4), stride=(4, 4))
            tmp[0][0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))
            tmp[0][0].bias = oldb
        del tmp[7]
        del tmp[6]
        self.backbone = tmp
        self.classiflow = torch.nn.Conv2d(384, 13, kernel_size=1)

        ks = (2, 1, 1)
        self.conv1 = torch.nn.Conv3d(10, 32, kernel_size=ks, stride=ks, padding=0)
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=ks, stride=ks, padding=0)
        self.conv3 = torch.nn.Conv3d(64, 92, kernel_size=(3, 3, 3))
        self.conv4 = torch.nn.Conv3d(92, 128, kernel_size=ks, stride=ks, padding=0)
        self.conv5 = torch.nn.Conv2d(256, 128, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(128, 128, kernel_size=3)

        self.merge1 = torch.nn.Conv2d(512, 256, kernel_size=1)
        self.merge2 = torch.nn.Conv2d(640, 256, kernel_size=1)
        self.merge3 = torch.nn.Conv2d(640, 128, kernel_size=1)

        self.decod1 = torch.nn.Conv2d(608, 224, kernel_size=1)
        self.decod2 = torch.nn.Conv2d(320, 224, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(320, 224, kernel_size=3, padding=1)
        self.decod4 = torch.nn.Conv2d(320, 224, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(320, 13, kernel_size=1)

        self.compress = torch.nn.Conv2d(128, 2, kernel_size=1)
        self.expand = torch.nn.Conv2d(2, 64, kernel_size=1)
        self.expand2 = torch.nn.Conv2d(13, 64, kernel_size=1)
        self.generate1 = torch.nn.Conv2d(64, 128, kernel_size=1)
        self.generate2 = torch.nn.Conv2d(128, 32, kernel_size=1)
        self.generate3 = torch.nn.Conv2d(32, 10, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forwardRGB(self, x):
        x = ((x / 255) - 0.5) / 0.25
        xm = torch.ones(x.shape[0], 1, 512, 512).cuda()
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[1](self.backbone[0](x))  # 96
        x = self.backbone[5](
            self.backbone[4](self.backbone[3](self.backbone[2](hr)))
        )  # 384
        plow = self.classiflow(x)
        plow = torch.nn.functional.interpolate(plow, size=(512, 512), mode="bilinear")

        return plow, x, hr

    def forwardSentinel(self, s):
        s = self.lrelu(self.conv1(s))
        s = self.lrelu(self.conv2(s))
        s = self.lrelu(self.conv3(s))
        s = self.lrelu(self.conv4(s))

        ss = s.mean(2)
        s, _ = s.max(2)
        s = torch.cat([s, ss], dim=1)

        s = self.lrelu(self.conv5(s))
        s = self.lrelu(self.conv6(s))
        s = self.lrelu(self.conv7(s))
        return s

    def forwardClassifier(self, x, hr, s):
        xs = torch.cat([x, s], dim=1)
        xs = self.lrelu(self.merge1(xs))
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge2(xs))
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge3(xs))

        f = torch.nn.functional.interpolate(xs, size=(128, 128), mode="bilinear")
        f1 = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        f2 = torch.cat([f1, f, hr], dim=1)
        f2 = self.lrelu(self.decod1(f2))
        f2 = torch.cat([f2, hr], dim=1)
        f2 = self.lrelu(self.decod2(f2))
        f2 = torch.cat([f2, hr], dim=1)
        f2 = self.lrelu(self.decod3(f2))
        f2 = torch.cat([f2, hr], dim=1)
        f2 = self.lrelu(self.decod4(f2))
        f2 = torch.cat([f2, hr], dim=1)
        p = self.classif(f2)

        return p, xs

    def forward(self, x, s, mode=1):
        assert 1 <= mode <= 4

        if mode == 1:
            plow, x, hr = self.forwardRGB(x)
            s = self.forwardSentinel(s)
            p, _ = self.forwardClassifier(x, hr, s)
            p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
            return p + 0.1 * plow

        if mode == 2:
            plow, _, _ = self.forwardRGB(x)
            return plow

        if mode >= 3:
            if mode == 3:
                with torch.no_grad():
                    plow, x, hr = self.forwardRGB(x)
            else:
                plow, x, hr = self.forwardRGB(x)
            p, xs = self.forwardClassifier(x, hr, self.forwardSentinel(s))

            xs = self.compress(xs)
            xs = torch.nn.functional.interpolate(xs, size=(40, 40), mode="bilinear")
            xs = self.expand(xs)
            ps = torch.nn.functional.interpolate(p, size=(40, 40), mode="bilinear")
            ps = self.expand2(ps)
            xs = ps * xs

            xs = self.lrelu(self.generate1(xs))
            xs = self.lrelu(self.generate2(xs))
            xs = self.generate3(xs)
            xs = xs * 0.1 + 0.9 * torch.clamp(xs, -1, 1)
            assert xs.shape[1:] == (10, 40, 40)

            loss = ((xs - s.mean(2)) ** 2).flatten().mean()
            tmp = (xs.unsqueeze(2) - s) ** 2
            assert tmp.shape[1:] == (10, 32, 40, 40)
            losses = torch.zeros(tmp.shape[0]).cuda()
            for i in range(tmp.shape[0]):
                losses[i] = min([tmp[i, :, j, :, :].mean() for j in range(32)])
            loss = loss + losses.mean()

            p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
            return p + 0.1 * plow, loss


if __name__ == "__main__":
    import os

    os.system("/d/achanhon/miniconda3/bin/python -u train.py")
    os.system("/d/achanhon/miniconda3/bin/python -u val.py")
    os.system("/d/achanhon/miniconda3/bin/python -u test.py")
    quit()
