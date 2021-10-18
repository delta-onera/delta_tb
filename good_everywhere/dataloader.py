import numpy
import os
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


class MiniWorld:
    def __init__(self, flag, custom=None, tilesize=128):
        assert flag in ["train", "test", "custom"]

        self.names = [
            "potsdam",
            "christchurch",
            "toulouse",
            "austin",
            "chicago",
            "kitsap",
            "tyrol-w",
            "vienna",
            "bruges",
            "Arlington",
            "Austin",
            "DC",
            "NewYork",
            "SanFrancisco",
            "Atlanta",
            "NewHaven",
            "Norfolk",
            "Seekonk",
        ]
        self.tilesize = tilesize

        if flag == "custom":
            self.names = custom
        else:
            if flag == "train":
                self.names = [s + "/train/" for s in self.names]
            else:
                self.names = [s + "/test/" for s in self.names]

        whereIam = os.uname()[1]
        if whereIam in ["super", "wdtim719z"]:
            self.root = "/data/miniworld/"
        if whereIam == "ldtis706z":
            self.root = "/media/achanhon/bigdata/data/miniworld/"
        if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
            self.root = "/scratch_ai4geo/miniworld/"

        self.data = {}
        for town in self.names:
            self.data[town] = cropextractor.CropExtractor(
                self.root + town, tilesize=tilesize
            )
        self.run = False

    def start():
        self.run = True
        for town in self.names:
            self.data[town].start()

    def getbatch(self, batchsize, batchpriority=None):
        assert self.run
        assert batchpriority is None or numpy.sum(batchpriority) > 0

        tilesize = self.tilesize
        if batchpriority is None:
            batchpriority = numpy.ones(self.names)
        batchpriority /= numpy.sum(batchpriority)

        batchchoice = numpy.random.choice(self.names, batchsize, p=batchpriority)

        x, y = torch.zeros(batchsize, 3, tilesize, tilesize), torch.zeros(
            batchsize, tilesize, tilesize
        )
        for i, name in enumerate(batchchoice):
            x[i], y[i] = self.data[name].getcrop()
        return x, y, batchchoice
