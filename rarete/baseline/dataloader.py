import os
import PIL
from PIL import Image
import numpy
import torch
import random
import queue
import threading


def symetrie(x, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x = numpy.transpose(x, axes=(1, 0, 2))
    if j == 1:
        x = numpy.flip(x, axis=1)
    if k == 1:
        x = numpy.flip(x, axis=0)
    return x.copy()


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


class Dataloader(threading.Thread):
    def __init__(self, paths, maxsize=10, batchsize=64):
        threading.Thread.__init__(self)
        self.isrunning = False

        self.maxsize = maxsize
        self.batchsize = batchsize
        self.paths = paths

    def getImages(self, i, torchformat=False):
        assert i < len(self.path)

        img1 = PIL.Image.open(self.paths[i] + "_1.png").convert("RGB").copy()
        img1 = numpy.uint8(numpy.asarray(img1))
        img2 = PIL.Image.open(self.paths[i] + "_2.png").convert("RGB").copy()
        img2 = numpy.uint8(numpy.asarray(img2))

        if torchformat:
            return pilTOtorch(img1), pilTOtorch(img2)
        else:
            return img1, img2

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def run(self):
        assert not self.isrunning
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        batchsize = self.batchsize

        while True:
            I = (torch.rand(self.batchsize) * len(self.paths)).long()
            flag = numpy.random.randint(0, 2, size=(self.batchsize, 3))
            batch = torch.zeros(batchsize, 6, 48, 48)
            for i in range(self.batchsize):
                img1, img2 = self.getImages(I[i], torchformat=False)
                img1, img2 = symetrie(img1, flag[i]), symetrie(img2, flag[i])
                img1, img2 = pilTOtorch(x), torch.Tensor(y)
                batch[i, 0:3], batch[i, 3:6] = img1, img2
            self.q.put(batch, block=True)


def getstdtraindataloader():
    root = "../preprocessing/build/"
    paths = [str(i) for i in range(2358)]
    paths = [paths[i] for i in range(len(paths)) if i % 4 < 2]
    paths = [root + path for path in paths]
    return Dataloader(paths)


def getstdtestdataloader():
    root = "../preprocessing/build/"
    paths = [str(i) for i in range(2358)]
    paths = [paths[i] for i in range(len(paths)) if i % 4 >= 2]
    paths = [root + path for path in paths]
    return Dataloader(paths)
