import os
import PIL
from PIL import Image
import rasterio
import numpy
import torch
import queue
import threading
import torchvision


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


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        y = numpy.transpose(y, axes=(1, 0))
        for u in range(x.shape[0]):
            x[u] = numpy.transpose(x[u], axes=(1, 0))
    if j == 1:
        y = numpy.flip(y, axis=0)
        for u in range(x.shape[0]):
            x[u] = numpy.flip(x[u], axis=0)
    if k == 1:
        y = numpy.flip(y, axis=1)
        for u in range(x.shape[0]):
            x[u] = numpy.flip(x[u], axis=1)
    return x.copy(), y.copy()


class CropExtractor(threading.Thread):
    def __init__(self, paths):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 161
        self.paths = paths

    def getName(self, i):
        return self.paths[i][2]

    def getImageAndLabel(self, i, torchformat=False):
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        y = PIL.Image.open(self.paths[i][1]).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.clip(numpy.nan_to_num(y) - 1, 0, 12)

        if torchformat:
            return torch.Tensor(x), torch.Tensor(y)
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        x = torch.zeros(batchsize, 5, 512, 512)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)

        while True:
            i = int(torch.rand(1) * len(self.paths))
            image, label = self.getImageAndLabel(i)
            flag = numpy.random.randint(0, 2, size=3)
            x, y = symetrie(image.copy(), label.copy(), flag)
            x, y = torch.Tensor(x), torch.Tensor(y)
            self.q.put((x, y), block=True)


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["1/4", "3/4"]
        self.root = root
        self.run = False
        self.paths = []

        domaines = os.listdir(root)
        for domaine in domaines:
            sousdomaines = os.listdir(root + domaine)
            for sousdomaines in sousdomaines:
                prefix = root + domaine + "/" + sousdomaines
                names = os.listdir(prefix + "/img")
                names = [name for name in names if ".aux" not in name]
                names = [name[4:] for name in names if "IMG_" in name]

                for name in names:
                    x = prefix + "/img/IMG_" + name
                    y = prefix + "/msk/MSK_" + name
                    self.paths.append((x, y, name))

        self.paths = sorted(self.paths)
        N = len(self.paths)
        if flag == "3/4":
            self.paths = [self.paths[i] for i in range(N) if i % 4 != 0]
        if flag == "1/4":
            self.paths = [self.paths[i] for i in range(N) if i % 4 == 0]

        self.data = CropExtractor(self.paths)

    def getImageAndLabel(self, i):
        return self.data.getImageAndLabel(i, torchformat=True)

    def getName(self, i):
        return self.data.getName(i)

    def getBatch(self, batchsize):
        assert self.run
        return self.data.getBatch(batchsize)

    def start(self):
        if not self.run:
            self.run = True
            self.data.start()


class UNET_EFFICIENTNET(torch.nn.Module):
    def __init__(self):
        super(UNET_EFFICIENTNET, self).__init__()
        self.f = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        with torch.no_grad():
            tmp = torch.cat([self.f[0][0].weight.clone()] * 2, dim=1)
            self.f[0][0] = torch.nn.Conv2d(
                6, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.f[0][0].weight = torch.nn.Parameter(tmp * 0.5)

        self.g1 = torch.nn.Conv2d(1504, 256, kernel_size=5, padding=2)
        self.g2 = torch.nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.g3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.g4 = torch.nn.Conv2d(2112, 256, kernel_size=5, padding=2)
        self.g5 = torch.nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.g6 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(2112, 13, kernel_size=1)

    def forward(self, x):
        b, ch, h, w = x.shape
        assert ch == 5 and w == 512 and h == 512
        padding = torch.ones(b, 1, h, w).cuda()

        x = ((x / 255) - 0.5) / 0.5
        x = torch.cat([x, padding], dim=1)

        z3 = self.f[3](self.f[2](self.f[1](self.f[0](x))))
        z4 = self.f[5](self.f[4](z3))
        z5 = self.f[8](self.f[7](self.f[6](z4)))

        z5 = torch.nn.functional.interpolate(z5, size=(32, 32), mode="bilinear")
        z = torch.cat([z4, z5], dim=1)
        z = torch.nn.functional.leaky_relu(self.g1(z))
        z = torch.nn.functional.leaky_relu(self.g2(z))
        z = torch.nn.functional.leaky_relu(self.g3(z))

        z4 = torch.nn.functional.interpolate(z4, size=(64, 64), mode="bilinear")
        z5 = torch.nn.functional.interpolate(z5, size=(64, 64), mode="bilinear")
        z = torch.nn.functional.interpolate(z, size=(64, 64), mode="bilinear")
        z = torch.cat([z, z3, z4, z5], dim=1)
        z = torch.nn.functional.leaky_relu(self.g4(z))
        z = torch.nn.functional.leaky_relu(self.g5(z))
        z = torch.nn.functional.leaky_relu(self.g6(z))

        z = torch.cat([z, z3, z4, z5], dim=1)
        z = self.classif(z)

        # z = torch.nn.functional.interpolate(z, size=(h, w), mode="bilinear")
        return z
