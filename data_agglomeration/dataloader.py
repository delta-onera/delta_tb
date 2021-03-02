import numpy as np


def safeuint8(x):
    x0 = np.zeros(x.shape, dtype=float)
    x255 = np.ones(x.shape, dtype=float) * 255
    x = np.maximum(x0, np.minimum(x.copy(), x255))
    return np.uint8(x)


def symetrie(x, y, i, j, k):
    if i == 1:
        x, y = np.transpose(x, axes=(1, 0, 2)), np.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    if k == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    return x.copy(), y.copy()


import os
import PIL
from PIL import Image

import torch
import torchvision


class SegSemDataset:
    def __init__(self, pathTOdata, vtflag):
        self.pathTOdata = pathTOdata
        self.vtflag = vtflag

    self.nbImages = 0
    while os.path.exists(
        self.pathTOdata + str(self.nbImages) + "_x.png"
    ) and os.path.exists(self.pathTOdata + str(self.nbImages) + "_y.png"):
        self.nbImages += 1

    nbbat, nbnonbat = 0, 0
    for i in range(self.nbImages):
        label = PIL.Image.open(self.pathTOdata + str(i) + "_y.png").convert("L").copy()
        label = np.asarray(label, dtype=np.uint8)  # warning wh swapping
        label = np.uint8(label != 0)
        nbbat += np.sum((label == 1).astype(int))
        nbnonbat += np.sum((label == 0).astype(int))

    self.balance = nbnonbat / nbbat

    ###
    ### get the hole image
    ### for test usage -- or in internal call for extracting crops
    def getImageAndLabel(self, i, innumpy=True):
        assert i < self.nbImages

        image = (
            PIL.Image.open(self.pathTOdata + str(i) + "_x.png").convert("RGB").copy()
        )
        image = np.asarray(image, dtype=np.uint8)  # warning wh swapping

        label = PIL.Image.open(self.pathTOdata + str(i) + "_y.png").convert("L").copy()
        label = np.asarray(label, dtype=np.uint8)  # warning wh swapping
        label = np.uint8(label != 0)

        if innumpy:
            return image, label
        else:
            image = torch.Tensor(np.transpose(image, axes=(2, 0, 1))).unsqueeze(0)
            return image, label

    ###
    ### get randomcrops + symetrie
    ### get train usage
    def getrawrandomtiles(self, nbtiles, tilesize):
        XY = []
        nbtilesperimage = nbtiles // self.images + 1

        # crop
        for name in range(self.images):
            image, label = self.getImageAndLabel(name)

            row = np.random.randint(
                0, image.shape[0] - tilesize - 2, size=nbtilesperimage
            )
            col = np.random.randint(
                0, image.shape[1] - tilesize - 2, size=nbtilesperimage
            )

            for i in range(nbtilesperimage):
                im = image[
                    row[i] : row[i] + tilesize, col[i] : col[i] + tilesize, :
                ].copy()
                mask = label[
                    row[i] : row[i] + tilesize, col[i] : col[i] + tilesize
                ].copy()
                XY.append((im, mask))

        # symetrie
        symetrieflag = np.random.randint(0, 2, size=(len(XY), 3))
        XY = [
            (symetrie(x, y, symetrieflag[i][0], symetrieflag[i][1], symetrieflag[i][2]))
            for i, (x, y) in enumerate(XY)
        ]
        return XY
