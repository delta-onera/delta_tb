import os
import PIL
from PIL import Image
import numpy
import torch
import queue
import threading
import torchvision


def confusion(y, z):
    cm = torch.zeros(2, 2).cuda()
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cm[a][b] = ((z == a).float() * (y == b).float()).sum()
    return cm


def numpyTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def perf(cm):
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=0), numpy.flip(y, axis=0)
    return x.copy(), y.copy()


class CropExtractor(threading.Thread):
    def __init__(self, path, tile=256):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 500
        self.tilesize = tile
        self.path = path

        self.NB = 0
        while os.path.exists(self.path + str(self.NB) + "_x.png"):
            self.NB += 1
        if self.NB == 0:
            print("wrong path", self.path)
            quit()

    def getImageAndLabel(self, i, torchformat=False):
        assert i < self.NB

        x = PIL.Image.open(self.path + str(i) + "_x.png").convert("RGB").copy()
        x = numpy.uint8(numpy.asarray(x))

        y = PIL.Image.open(self.path + str(i) + "_y.png").convert("L").copy()
        y = numpy.uint8(numpy.asarray(y) != 0)

        if torchformat:
            return numpyTOtorch(x), torch.Tensor(y)
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, self.tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize

        while True:
            for i in range(self.NB):
                image, label = self.getImageAndLabel(i)

                ntile = 50
                RC = numpy.random.rand(ntile, 2)
                flag = numpy.random.randint(0, 2, size=(ntile, 3))
                for j in range(ntile):
                    r = int(RC[j][0] * (image.shape[0] - tilesize - 2))
                    c = int(RC[j][1] * (image.shape[1] - tilesize - 2))
                    im = image[r : r + tilesize, c : c + tilesize, :]
                    mask = label[r : r + tilesize, c : c + tilesize]
                    x, y = symetrie(im.copy(), mask.copy(), flag[j])
                    x, y = numpyTOtorch(x), torch.Tensor(y)
                    self.q.put((x, y), block=True)


class Mobilenet(torch.nn.Module):
    def __init__(self):
        super(Mobilenet, self).__init__()
        self.backend = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights="DEFAULT"
        )
        self.backend.classifier.low_classifier = torch.nn.Conv2d(40, 2, kernel_size=1)
        self.backend.classifier.high_classifier = torch.nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25
        return self.backend(x)["out"]


class Deeplab(torch.nn.Module):
    def __init__(self):
        super(Deeplab, self).__init__()
        self.backend = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DEFAULT"
        )
        self.backend.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25
        return self.backend(x)["out"]


class GlobalLocal(torch.nn.Module):
    def __init__(self):
        super(GlobalLocal, self).__init__()
        self.backbone = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        self.compress = torch.nn.Conv2d(1280, 32, kernel_size=1)
        self.classiflow = torch.nn.Conv2d(1280, 2, kernel_size=1)

        self.local1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.local2 = torch.nn.Conv2d(99, 32, kernel_size=5, padding=2)
        self.local3 = torch.nn.Conv2d(99, 32, kernel_size=5, padding=2)
        self.local4 = torch.nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.local5 = torch.nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.classifhigh = torch.nn.Conv2d(32, 2, kernel_size=1)

    def forwardglobal(self, x):
        x = 2 * (x / 255) - 1
        x = torch.nn.functional.interpolate(
            x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="bilinear"
        )
        return torch.nn.functional.leaky_relu(self.backbone(x))

    def forwardlocal(self, x, f):
        z = self.local1(x)
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local2(z))
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local3(z))

        zz = torch.nn.functional.max_pool2d(z, kernel_size=3, stride=1, padding=1)
        zz = torch.nn.functional.relu(100 * z - 99 * zz)

        z = torch.cat([z, zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local4(z))
        z = torch.cat([z, z * zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local5(z))
        return self.classifhigh(z / 100)

    def forward(self, x, firsttrainstep=False):
        if firsttrainstep:
            with torch.no_grad():
                f = self.forwardglobal(x)
        else:
            f = self.forwardglobal(x)

        z = self.classiflow(f)
        z = torch.nn.functional.interpolate(
            z, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )

        f = torch.nn.functional.leaky_relu(self.compress(f))
        f = torch.nn.functional.interpolate(
            f, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        return self.forwardlocal(x, f) + z * 0.1
