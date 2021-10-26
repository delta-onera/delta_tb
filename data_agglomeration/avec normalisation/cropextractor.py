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


def normalize(image):
    if len(im.shape) == 2:
        allvalues = []
        for row in range(0, im.shape[0] - 100, 300):
            for col in range(0, im.shape[1] - 100, 300):
                allvalues += list(im[row : row + 100, col : col + 100].flatten())

        allvalues = [v for v in allvalues if v >= 2]

        allvalues = sorted(allvalues)
        n = len(allvalues)
        allvalues = allvalues[0 : int(98 * n / 100)]
        allvalues = allvalues[int(2 * n / 100) :]

        n = len(allvalues)
        k = n // 255
        pivot = [0] + [allvalues[i] for i in range(0, n, k)]
        assert len(pivot) >= 255

        out = np.zeros(im.shape, dtype=int)
        for i in range(1, 255):
            print(i)
            out = np.maximum(out, np.uint8(im > pivot[i]) * i)

        return safeuint8(out)

    else:
        output = im.copy()
        for i in range(im.shape[2]):
            output[:, :, i] = normalizehistogram(im[:, :, i])
        return output


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
