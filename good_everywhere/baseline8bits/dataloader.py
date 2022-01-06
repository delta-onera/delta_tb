import os
import sys
import numpy
import PIL
from PIL import Image
import torch
import random
import cropextractor


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


class MiniWorld:
    def __init__(self, flag, custom=None, tilesize=128):
        assert flag in ["train", "test", "custom"]

        whereIam = os.uname()[1]
        if whereIam == "wdtim719z":
            self.root = "/data/miniworld/"
        if whereIam == "ldtis706z":
            self.root = "/media/achanhon/bigdata/data/miniworld/"
        if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
            self.root = "/scratchf/miniworld/"

        self.cities = os.listdir(self.root)
        self.tilesize = tilesize

        if flag == "custom":
            self.cities = custom
        else:
            if flag == "train":
                self.cities = [s + "/train/" for s in self.cities]
            else:
                self.cities = [s + "/test/" for s in self.cities]

        print("loading data from", self.cities)

        self.data = {}
        for city in self.cities:
            if flag != "test":
                self.data[city] = cropextractor.CropExtractor(
                    self.root + city, tilesize=tilesize
                )
            else:
                self.data[city] = cropextractor.CropExtractor(
                    self.root + city, maxsize=0, tilesize=tilesize
                )
        self.run = False

    def start(self):
        assert not self.run
        self.run = True
        for city in self.cities:
            self.data[city].start()

    def getbatch(self, batchsize):
        assert self.run

        tilesize = self.tilesize
        priority = numpy.ones(len(self.cities))
        priority /= numpy.sum(priority)

        batchchoice = numpy.random.choice(len(self.cities), batchsize, p=priority)

        x = torch.zeros(batchsize, 3, tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.data[self.cities[batchchoice[i]]].getCrop()
        return x, y, batchchoice
