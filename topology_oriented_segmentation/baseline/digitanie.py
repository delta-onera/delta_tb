import os
import sys
import numpy
import PIL
from PIL import Image
import rasterio
import torch
import random


def maxpool(y, size):
    if len(y.shape) == 2:
        yy == y.unsqueeze(0).float()
    else:
        yy = y.float()

    ks = 2 * size + 1
    yyy = torch.nn.functional.max_pool2d(yy, kernel_size=ks, stride=1, padding=size)

    if len(y.shape) == 2:
        return yyy[0]
    else:
        return yyy


def minpool(y, size):
    yy = 1 - y  # 0->1,1->0
    yyy = maxpool(1 - y, size=size)  # 0 padding does not change the pooling
    return 1 - yyy


def isborder(y, size=2):
    y1 = (y == 1).float()
    y1pool = minpool(y1)
    y1boder = y1 * (y1pool == 0).float()

    y0 = (y == 0).float()
    y0pool = maxpool(y0)
    y0boder = y0 * (y0pool == 1).float()

    return ((y1boder + y0boder) > 0).float()


def confusion(y, z):
    D = 1 - isborder(y)
    cm = torch.zeros(2, 2).cuda()
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cm[a][b] = torch.sum((z == a).float() * (y == b).float() * D)
    return cm


def perf(cm):
    if len(cm.shape) == 2:
        accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
        iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
        iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
        return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))
    else:
        out = torch.zeros(cm.shape[0] + 1, 4)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        out[-1] = perf(torch.sum(cm, dim=0))
        return out


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def torchTOpil(x):
    return numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))


def resize(image, label):
    size = (image.size[0] // 2, image.size[1] // 2)
    image = image.resize(size, PIL.Image.BILINEAR)
    label = label.resize(size, PIL.Image.NEAREST)
    return image, label


def resizenumpy(image, label):
    image = PIL.Image.fromarray(numpy.uint8(image))
    label = PIL.Image.fromarray(numpy.uint8(label))
    image, label = resize(image=image, label=label)
    return numpy.asarray(image), numpy.asarray(label)


########################################################################
######################### PRIVATE CNES DATASET #########################


def digitanie_name():
    return ["Biarritz", "Strasbourg", "Paris"]


class DIGITANIE:
    def __init__(self, name, path):
        print("DIGITANIE", name)
        self.path = path
        self.name = name

        tmp = path + name + "/" + name.lower() + "_tuile_"
        self.NB = 0
        while os.path.exists(tmp + str(self.NB + 1) + "_c.tif"):
            self.NB += 1

        if self.NB == 0:
            print("wrong path", tmp)
            quit()

    def getImageAndLabel(self, i):
        assert i < self.NB

        tmp = self.path + self.name + "/" + self.name.lower() + "_tuile_"
        xpath = tmp + str(i + 1) + "_img_c.tif"
        ypath = tmp + str(i + 1) + "_c.tif"

        with rasterio.open(xpath) as src:
            r = numpy.int16(src.read(1) * 255)
            g = numpy.int16(src.read(2) * 255)
            b = numpy.int16(src.read(3) * 255)
            x = numpy.stack([r, g, b], axis=-1)

        y = PIL.Image.open(ypath).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.uint8((y[:, :] == 4))

        return resizenumpy(x, y)


class DIGITANIE_TOULOUSE:
    def __init__(self, path):
        print("DIGITANIE_TOULOUSE")
        self.path = path

        self.files = [
            ("tlse_arenes_c.tif", "tlse_arenes_img_c.tif"),
            ("tlse_bagatelle_c.tif", "tlse_bagatelle_img_c.tif"),
            ("tlse_cepiere_c.tif", "tlse_cepiere_img_c.tif"),
            ("tlse_empalot_c.tif", "tlse_empalot_img_c.tif"),
            ("tlse_mirail_c.tif", "tlse_mirail_img_c.tif"),
            ("tlse_montaudran_c.tif", "tlse_montaudran_img_c.tif"),
            ("tlse_zenith_c.tif", "tlse_zenith_img_c.tif"),
        ]
        self.NB = len(self.files)

    def getImageAndLabel(self, i):
        assert i < self.NB

        with rasterio.open(self.path + self.files[i][1]) as src:
            r = numpy.int16(src.read(1) * 255)
            g = numpy.int16(src.read(2) * 255)
            b = numpy.int16(src.read(3) * 255)
            x = numpy.stack([r, g, b], axis=-1)

        y = PIL.Image.open(self.path + self.files[i][0]).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.uint8((y[:, :] == 4))

        return resizenumpy(x, y)


######################### PRIVATE CNES DATASET #########################
########################################################################


class DigitanieALL:
    def __init__(self, names=None, path="/scratchf/PRIVATE/DIGITANIE/"):
        if names is not None:
            self.cities = names
        else:
            self.cities = digitanie_name() + ["digitanie_toulouse"]

        self.data = {}
        self.NB = {}
        self.normalization = {}
        for name in self.cities:
            if name in digitanie_name():
                self.data[name] = DIGITANIE(name, path)
            if name == "digitanie_toulouse":
                self.data[name] = DIGITANIE_TOULOUSE(path + "Toulouse/")

            self.NB[name] = self.data[name].NB

    def getImageAndLabel(self, city, i, torchformat=False):
        x, y = self.data[city].getImageAndLabel(i)

        if torchformat:
            return pilTOtorch(x), torch.Tensor(y)
        else:
            return x, y
