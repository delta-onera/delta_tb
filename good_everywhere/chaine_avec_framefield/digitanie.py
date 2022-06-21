import os
import sys
import numpy
import PIL
from PIL import Image
import rasterio
import torch
import random


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def torchTOpil(x):
    return numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))


def distancetransform(y, size):
    yy = 2.0 * y.unsqueeze(0) - 1
    yyy = torch.nn.functional.avg_pool2d(
        yy, kernel_size=2 * size + 1, stride=1, padding=size
    )
    D = 1.0 - 0.5 * (yy - yyy).abs()
    return D[0]


def perf(cm):
    if len(cm.shape) == 2:
        accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
        iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
        iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
        return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))
    else:
        out = torch.zeros(cm.shape[0], 4)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        return out


########################################################################
######################### PRIVATE CNES DATASET #########################


def digitanie_name():
    return ["Biarritz", "Montpellier", "Paris", "Strasbourg", "Toulouse"]


class DIGITANIE:
    def __init__(self, name, path):
        print("DIGITANIE", name)
        self.path = path
        self.name = name

        tmp = path + name + "/" + name.lower() + "_tuile_"
        for i in range(10):
            assert os.path.exists(tmp + str(self.NB + 1) + "_c.tif")

    def getImageAndLabel(self, i):
        assert i < 10

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

        return x, y


######################### PRIVATE CNES DATASET #########################
########################################################################


class DigitanieALL:
    def __init__(self, names=None, path="/scratchf/PRIVATE/DIGITANIE/"):
        self.cities = names

        for name in self.cities:
            self.data[name] = DIGITANIE(name, path)

    def getImageAndLabel(self, city, i, torchformat=False):
        x, y = self.data[city].getImageAndLabel(i)

        if torchformat:
            return pilTOtorch(x), torch.Tensor(y)
        else:
            return x, y
