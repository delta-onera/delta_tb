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


def distancetransform(y, size=4):
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
        return torch.Tensor((iou0 + iou1, accu))
    else:
        out = torch.zeros(cm.shape[0], 2)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        return out


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

        return x, y


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

        return x, y


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
