import os
import threading
import queue
import PIL
from PIL import Image
import numpy
import torch
import random


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    return x.copy(), y.copy()


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def torchTOpil(x):
    return numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))


class CropExtractor(threading.Thread):
    def __init__(self, path, maxsize=1000, tilesize=128):
        threading.Thread.__init__(self)
        self.path = path
        self.NB = 0
        while os.path.exists(self.path + str(self.NB) + "_x.png"):
            self.NB += 1

        if self.NB == 0:
            print("wrong path", self.path)
            quit()

        if maxsize > 0:
            self.tilesize = tilesize
            self.q = queue.Queue(maxsize=maxsize)
        else:
            self.tilesize = None

    def getImageAndLabel(self, i, torchformat=False):
        assert i < self.NB

        image = PIL.Image.open(self.path + str(i) + "_x.png").convert("RGB").copy()
        image = numpy.uint8(numpy.asarray(image))

        label = PIL.Image.open(self.path + str(i) + "_y.png").convert("L").copy()
        label = numpy.uint8(numpy.asarray(label))
        label = numpy.uint8(label != 0)

        if torchformat:
            return pilTOtorch(image), torch.Tensor(label)
        else:
            return image, label

    def getCrop(self):
        return self.q.get(block=True)

    ###############################################################

    def run(self):
        assert self.tilesize is not None
        tilesize = self.tilesize

        while True:
            I = [i for i in range(self.NB)]
            random.shuffle(I)
            for i in I:
                image, label = self.getImageAndLabel(i, torchformat=False)

                RC = numpy.random.rand(16, 2)
                flag = numpy.random.randint(0, 2, size=(16, 3))
                for j in range(16):
                    r = int(RC[j][0] * (image.shape[0] - tilesize - 2))
                    c = int(RC[j][1] * (image.shape[1] - tilesize - 2))
                    im = image[r : r + tilesize, c : c + tilesize, :]
                    mask = label[r : r + tilesize, c : c + tilesize]
                    x, y = symetrie(im.copy(), mask.copy(), flag[j])
                    x, y = pilTOtorch(x), torch.Tensor(y)
                    self.q.put((x, y), block=True)
