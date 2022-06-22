import os
import sys

########################################################################
######################### PRIVATE CNES DATASET #########################

import numpy
import PIL
from PIL import Image
import rasterio
import torch
import random
import util


def writeImage(source, newdata, target):
    with rasterio.open(source, "r") as src:
        profile = src.profile

    with rasterio.open(target, "w", **profile) as trgt:
        if len(newdata.shape) == 2:
            trgt.write(newdata[:, :], 1)
        else:
            trgt.write(newdata[0, :, :], 1)
            trgt.write(newdata[1, :, :], 2)
            trgt.write(newdata[2, :, :], 3)


class DIGITANIE:
    def __init__(self, name, path):
        self.names = ["Biarritz", "Montpellier", "Paris", "Strasbourg", "Toulouse"]
        assert name in self.names

        print("DIGITANIE", name)
        self.path = path
        self.name = name
        tmp = self.path + self.name + "/" + self.name.lower() + "_tuile_"
        for i in range(10):
            assert os.path.exists(tmp + str(i + 1) + "_img_normalized.tif")
            assert os.path.exists(tmp + str(i + 1) + ".tif")

    def getPath(self, i):
        assert i < 10
        tmp = self.path + self.name + "/" + self.name.lower() + "_tuile_"
        xpath = tmp + str(i + 1) + "_img_normalized.tif"
        ypath = tmp + str(i + 1) + ".tif"
        return xpath, ypath

    def getImageAndLabel(self, i):
        xpath, ypath = self.getPath(i)

        with rasterio.open(xpath, "r") as src:
            r = numpy.int16(src.read(1) * 255)
            g = numpy.int16(src.read(2) * 255)
            b = numpy.int16(src.read(3) * 255)
        x = numpy.stack([r, g, b], axis=-1)

        with rasterio.open(xpath, "r") as src:
            y = numpy.int16(src.read(1))
        y = numpy.uint8((y[:, :] == 4))

        return x, y


class DigitanieALL:
    def __init__(self, names=None, path="/scratchf/PRIVATE/DIGITANIE/"):
        self.cities = names
        for name in self.cities:
            self.data[name] = DIGITANIE(name, path)

    def getPath(self, city, i):
        return self.data[city].getPath(i)

    def getImageAndLabel(self, city, i, torchformat=False):
        x, y = self.data[city].getImageAndLabel(i)

        if torchformat:
            return util.pilTOtorch(x), torch.Tensor(y)
        else:
            return x, y


######################### PRIVATE CNES DATASET #########################
########################################################################
