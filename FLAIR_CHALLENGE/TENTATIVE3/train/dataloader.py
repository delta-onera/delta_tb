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
            x = torch.Tensor(x) / 255

        y = PIL.Image.open(self.paths[i][1]).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.clip(numpy.nan_to_num(y) - 1, 0, 12)
        y = torch.Tensor(y)

        x = [x]
        for mode in ["model1/", "model2/", "model3/", "model4/"]:
            path = "../preparetrain/build/"
            tmp = path + mode + self.getName(i) + ".pth"
            tmp = torch.load(tmp, map_location=torch.device("cpu"))
            tmp = tmp.unsqueeze(0).float()
            tmp = torch.nn.functional.interpolate(tmp, size=(512, 512), mode="bilinear")
            x.append(tmp[0] / 15)

        x = torch.cat(x, dim=0)
        assert x.shape == (57, 512, 512)
        return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        x = torch.zeros(batchsize, 57, 512, 512)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)

        while True:
            i = int(torch.rand(1) * len(self.paths))
            x, y = self.getImageAndLabel(i)
            # flag = numpy.random.randint(0, 2, size=3)
            # x, y = symetrie(image.copy(), label.copy(), flag)
            # x, y = torch.Tensor(x), torch.Tensor(y)
            self.q.put((x, y), block=True)


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["1/4"]
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


class FUSION(torch.nn.Module):
    def __init__(self):
        super(FUSION, self).__init__()
        self.f1 = torch.nn.Conv2d(57, 57, kernel_size=17, padding=8)
        self.f2 = torch.nn.Conv2d(171, 57, kernel_size=9, padding=4)
        self.f3 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
        self.f4 = torch.nn.Conv2d(171, 196, kernel_size=1)
        self.f5 = torch.nn.Conv2d(196, 13, kernel_size=1)

    def forward(self, x):
        z = torch.nn.functional.leaky_relu(self.f1(x))
        z = torch.cat([x, z, x * z * 0.1], dim=1)
        z = torch.nn.functional.leaky_relu(self.f2(z))
        z = torch.cat([x, z, x * z * 0.1], dim=1)
        z = torch.nn.functional.leaky_relu(self.f3(z))
        x = torch.cat([x, z, x * z * 0.1], dim=1)
        x = torch.nn.functional.leaky_relu(self.f4(x))
        return self.f5(x)
