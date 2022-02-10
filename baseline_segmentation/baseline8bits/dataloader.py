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
    tmp = torch.ones(D.shape)
    D = torch.minimum(tmp, (y == 0).float() * 0.3 + D)
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
    def __init__(self, flag, tilesize=128, custom=None):
        assert flag in ["train", "test"]
        self.tilesize = tilesize

        whereIam = os.uname()[1]
        if whereIam == "wdtim719z":
            self.root = "/data/miniworld/"
        if whereIam == "ldtis706z":
            self.root = "/media/achanhon/bigdata/data/miniworld/"
        if whereIam in ["calculon", "astroboy", "flexo", "bender", "baymax"]:
            self.root = "/scratchf/miniworld/"

        existingcities = os.listdir(self.root)
        if flag != "custom":
            expectedcities = [
                "potsdam",
                "christchurch",
                "bruges",
                "pologne",
                "Arlington",
                "NewHaven",
                "Norfolk",
                "Seekonk",
                "Atlanta",
                "Austin",
                "DC",
                "NewYork",
                "SanFrancisco",
                "chicago",
                "kitsap",
                "austin",
                "tyrol-w",
                "vienna",
                "rio",
            ]
            for city in expectedcities:
                if city not in existingcities:
                    print("missing city", city)
                    quit()
            self.cities = expectedcities

            if flag == "train":
                self.cities = [s + "/train/" for s in self.cities]
            else:
                self.cities = [s + "/test/" for s in self.cities]
        else:
            self.cities = custom

        print("loading data from", self.cities)

        self.data = {}
        self.run = False
        for city in self.cities:
            self.data[city] = cropextractor.CropExtractor(
                self.root + city, tilesize=tilesize
            )

    def start(self):
        if not self.run:
            self.run = True
            for city in self.cities:
                self.data[city].start()

    def getbatch(self, batchsize):
        assert self.run

        tilesize = self.tilesize
        priority = numpy.asarray([2, 4, 2, 4] + [1] * 9 + [2] * 5 + [3])
        priority = numpy.float32(priority) / numpy.sum(priority)
        batchchoice = numpy.random.choice(len(self.cities), batchsize, p=priority)

        x = torch.zeros(batchsize, 3, tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.data[self.cities[batchchoice[i]]].getCrop()
        return x, y, torch.Tensor(batchchoice)
