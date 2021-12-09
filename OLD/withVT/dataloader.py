import numpy as np
import os


def symetrie(x, y, i, j, k):
    if i == 1:
        x, y = np.transpose(x, axes=(1, 0, 2)), np.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    if k == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    return x.copy(), y.copy()


import PIL
from PIL import Image
import torch
import random


def distancetransform(y, size=4):
    yy = 2.0 * y.unsqueeze(0) - 1
    yyy = torch.nn.functional.avg_pool2d(
        yy, kernel_size=2 * size + 1, stride=1, padding=size
    )
    D = 1.0 - 0.5 * (yy - yyy).abs()
    return D[0]


class SegSemDataset:
    def __init__(self, pathTOdata, FLAGinteractif=33, batchsize=4):
        self.pathTOdata = pathTOdata
        self.FLAGinteractif = FLAGinteractif
        self.nbImages = 0
        self.batchsize = batchsize
        while os.path.exists(self.mypath(self.nbImages, True)):
            self.nbImages += 1
        if self.nbImages == 0:
            print("wrong path", self.pathTOdata)
            quit()

    def mypath(self, i, flag):
        if flag:
            return self.pathTOdata + str(i) + "_x.png"
        else:
            return self.pathTOdata + str(i) + "_y.png"

    def getImageAndLabel(self, i):
        assert i < self.nbImages

        image = PIL.Image.open(self.mypath(i, True)).convert("RGB").copy()
        image = np.uint8(np.asarray(image))

        label = PIL.Image.open(self.mypath(i, False)).convert("L").copy()
        label = np.uint8(np.asarray(label))
        label = np.uint8(label != 0)
        return image, label

    def topytorch(self, x, y):
        xx = torch.zeros(4, x.shape[0], x.shape[1])
        xx[0:3, :, :] = torch.Tensor(np.transpose(x, axes=(2, 0, 1)))
        if random.randint(0, 99) >= self.FLAGinteractif:
            xx[3, 0:64, :] = 2 * torch.Tensor(y[0:64, :]) - 1
        return torch.Tensor(xx)

    def finalize(self, XY):
        flag = np.random.randint(0, 2, size=(len(XY), 3))
        XY = [
            (symetrie(x, y, flag[i][0], flag[i][1], flag[i][2]))
            for i, (x, y) in enumerate(XY)
        ]

        Y = torch.stack([torch.from_numpy(y).long().cpu() for x, y in XY])
        X = torch.stack([self.topytorch(x, y).cpu() for x, y in XY])
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batchsize, shuffle=True, num_workers=2
        )
        return dataloader

    def getFrozenTiles(self, tilesize=256, stride=128):
        XY = []
        for name in range(self.nbImages):
            image, label = self.getImageAndLabel(name)

            for row in range(0, image.shape[0] - tilesize - 2, stride):
                for col in range(0, image.shape[1] - tilesize - 2, stride):
                    im = image[row : row + tilesize, col : col + tilesize, :]
                    mask = label[row : row + tilesize, col : col + tilesize]
                    XY.append((im.copy(), mask.copy()))

        return self.finalize(XY)

    def getrawrandomtiles(self, nbtiles=10000, tilesize=256):
        XY = []
        NB = int(nbtiles / self.nbImages + 1)

        for name in range(self.nbImages):
            image, label = self.getImageAndLabel(name)

            row = np.random.randint(0, image.shape[0] - tilesize - 2, size=NB)
            col = np.random.randint(0, image.shape[1] - tilesize - 2, size=NB)

            for i in range(NB):
                im = image[row[i] : row[i] + tilesize, col[i] : col[i] + tilesize, :]
                mask = label[row[i] : row[i] + tilesize, col[i] : col[i] + tilesize]
                XY.append((im.copy(), mask.copy()))

        return self.finalize(XY)
