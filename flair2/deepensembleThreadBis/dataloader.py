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
    def __init__(self, root="/d/achanhon/FLAIR_2/"):
        threading.Thread.__init__(self)
        self.root = root
        self.isrunning = False
        self.paths = torch.load(root + "alltestpaths.pth")

        self.q = queue.Queue(maxsize=100)

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

        return torch.Tensor(x), sen

    def asynchroneGet(self):
        assert self.isrunning
        return self.q.get(block=True)

    def run(self):
        assert self.isrunning == False
        self.isrunning = True

        for name in self.paths:
            x, s = self.get(name)
            self.q.put((name, x, s), block=True)


import PIL
from PIL import Image


def number6(i):
    s = str(i)
    while len(s) < 6:
        s = "0" + s
    return s


class ImageWritter(threading.Thread):
    def __init__(self, N):
        threading.Thread.__init__(self)
        self.q = queue.Queue(maxsize=10000)
        self.N = N

    def asynchronePush(self, path, image):
        self.q.put((path, image), block=True)

    def run(self):
        for n in range(self.N):
            path, image = self.q.get(block=True)
            image = PIL.Image.fromarray(image)
            image.save(path, compression="tiff_lzw")


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

        ks = (2, 1, 1)
        self.conv1 = torch.nn.Conv3d(10, 32, kernel_size=ks, stride=ks, padding=0)
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=ks, stride=ks, padding=0)
        self.conv3 = torch.nn.Conv3d(64, 92, kernel_size=(3, 3, 3))
        self.conv4 = torch.nn.Conv3d(92, 128, kernel_size=ks, stride=ks, padding=0)
        self.conv5 = torch.nn.Conv2d(256, 160, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(160, 160, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(160, 160, kernel_size=3)

        self.merge1 = torch.nn.Conv2d(320, 512, kernel_size=1)
        self.merge2 = torch.nn.Conv2d(672, 768, kernel_size=1)
        self.merge3 = torch.nn.Conv2d(928, 160, kernel_size=1)

        self.decod1 = torch.nn.Conv2d(368, 208, kernel_size=1)
        self.decod2 = torch.nn.Conv2d(416, 208, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(416, 208, kernel_size=3, padding=1)
        self.decod4 = torch.nn.Conv2d(416, 304, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(512, 13, kernel_size=1)

        self.compress = torch.nn.Conv2d(320, 2, kernel_size=1)
        self.expand = torch.nn.Conv2d(2, 64, kernel_size=1)
        self.expand2 = torch.nn.Conv2d(13, 64, kernel_size=1)
        self.generate1 = torch.nn.Conv2d(64, 128, kernel_size=1)
        self.generate2 = torch.nn.Conv2d(128, 32, kernel_size=1)
        self.generate3 = torch.nn.Conv2d(32, 10, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forwardRGB(self, x):
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        xm = xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.5
        x = x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[2](self.backbone[1](self.backbone[0](x)))
        x = self.backbone[5](self.backbone[4](self.backbone[3](hr)))
        plow = self.classiflow(x).float()
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
        xs = self.lrelu(self.merge3(xs)).float()

        x = x.float()
        f1 = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        f2 = torch.nn.functional.interpolate(xs, size=(128, 128), mode="bilinear")
        f1, f2 = f1.to(dtype=hr.dtype), f2.to(dtype=hr.dtype)
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

        return p, torch.cat([x, xs], dim=1)

    def forward(self, x, s):
        plow, x, hr = self.forwardRGB(x)
        s = self.forwardSentinel(s)
        p, _ = self.forwardClassifier(x, hr, s)
        p = p.float()
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
        return p + 0.1 * plow


class MyNet4(torch.nn.Module):
    def __init__(self):
        super(MyNet4, self).__init__()
        tmp = torchvision.models.swin_s(weights="DEFAULT").features
        del tmp[6:]
        with torch.no_grad():
            old = tmp[0][0].weight / 2
            tmp[0][0] = torch.nn.Conv2d(6, 96, kernel_size=4, stride=4)
            tmp[0][0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))
        self.vit = tmp
        self.classiflow = torch.nn.Conv2d(384, 13, kernel_size=1)

        ks = (2, 1, 1)
        self.conv1 = torch.nn.Conv3d(10, 32, kernel_size=ks, stride=ks, padding=0)
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=ks, stride=ks, padding=0)
        self.conv3 = torch.nn.Conv3d(64, 92, kernel_size=(3, 3, 3))
        self.conv4 = torch.nn.Conv3d(92, 128, kernel_size=ks, stride=ks, padding=0)
        self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(256, 256, kernel_size=3)

        self.merge1 = torch.nn.Conv2d(640, 640, kernel_size=1)
        self.merge2 = torch.nn.Conv2d(1024, 640, kernel_size=1)
        self.merge3 = torch.nn.Conv2d(1024, 128, kernel_size=1)

        self.decod1 = torch.nn.Conv2d(608, 128, kernel_size=1)
        self.decod2 = torch.nn.Conv2d(608, 128, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(608, 128, kernel_size=3, padding=1)
        self.decod4 = torch.nn.Conv2d(608, 128, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(608, 13, kernel_size=1)

        self.compress = torch.nn.Conv2d(128, 2, kernel_size=1)
        self.expand = torch.nn.Conv2d(2, 64, kernel_size=1)
        self.expand2 = torch.nn.Conv2d(13, 64, kernel_size=1)
        self.generate1 = torch.nn.Conv2d(64, 128, kernel_size=1)
        self.generate2 = torch.nn.Conv2d(128, 32, kernel_size=1)
        self.generate3 = torch.nn.Conv2d(32, 10, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forwardRGB(self, x):
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        xm = xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.25
        x = x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        hr = self.vit[0:2](x)
        x = self.vit[2:](hr)

        hr = torch.transpose(hr, 2, 3)
        hr = torch.transpose(hr, 1, 2)
        x = torch.transpose(x, 2, 3)
        x = torch.transpose(x, 1, 2)

        plow = self.classiflow(x).float()
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
        xs = self.lrelu(self.merge3(xs)).float()

        f = torch.nn.functional.interpolate(xs, size=(128, 128), mode="bilinear")
        x = x.float()
        x = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        x, f = x.to(dtype=hr.dtype), f.to(dtype=hr.dtype)
        f = torch.cat([f, x, hr], dim=1)
        f = self.lrelu(self.decod1(f))
        f = torch.cat([f, x, hr], dim=1)
        f = self.lrelu(self.decod2(f))
        f = torch.cat([f, x, hr], dim=1)
        f = self.lrelu(self.decod3(f))
        f = torch.cat([f, x, hr], dim=1)
        f = self.lrelu(self.decod4(f))
        f = torch.cat([f, x, hr], dim=1)
        p = self.classif(f)

        return p, xs

    def forward(self, x, s):
        plow, x, hr = self.forwardRGB(x)
        s = self.forwardSentinel(s)
        p, _ = self.forwardClassifier(x, hr, s)
        p = p.float()
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
        return p + 0.1 * plow
    

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
        xm = torch.ones(x.shape[0], 1, 512, 512).cuda()
        xm = xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.25
        x = x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[1](self.backbone[0](x))  # 96
        x = self.backbone[5](
            self.backbone[4](self.backbone[3](self.backbone[2](hr)))
        )  # 384
        plow = self.classiflow(x).float()
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
        xs = self.lrelu(self.merge3(xs)).float()

        f = torch.nn.functional.interpolate(xs, size=(128, 128), mode="bilinear")
        f1 = torch.nn.functional.interpolate(x.float(), size=(128, 128), mode="bilinear")
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

        return p

    def forward(self, x, s):
            plow, x, hr = self.forwardRGB(x)
            s = self.forwardSentinel(s)
            p = self.forwardClassifier(x, hr, s).float()
            p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
            return p + 0.1 * plow



if __name__ == "__main__":
    import os

    os.system("rm -rf build")
    os.system("mkdir build")
    os.system("/d/achanhon/miniconda3/bin/python -u test.py")
    quit()
