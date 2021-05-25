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
import random


def getindexeddata():
    whereIam = os.uname()[1]

    if whereIam in ["super", "wdtim719z"]:
        root = "/data/miniworld/"

    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        root = "/scratch_ai4geo/miniworld/"

    availabledata = [
        "potsdam",
        "christchurch",
        "toulouse",
        # "paris",
        "austin",
        "chicago",
        "kitsap",
        "tyrol-w",
        "vienna",
        # "vegas",
        # "shanghai",
        # "khartoum",
        "bruges",
        # "rio",
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

    return root, availabledata


class SegSemDataset:
    def __init__(self, pathTOdata):
        self.pathTOdata = pathTOdata

        self.nbImages = 0
        while os.path.exists(
            self.pathTOdata + str(self.nbImages) + "_x.png"
        ) and os.path.exists(self.pathTOdata + str(self.nbImages) + "_y.png"):
            self.nbImages += 1
        if self.nbImages == 0:
            print("wrong path", self.pathTOdata)
            quit()

        self.nbbat, self.nbnonbat = 0, 0
        for i in range(self.nbImages):
            label = (
                PIL.Image.open(self.pathTOdata + str(i) + "_y.png").convert("L").copy()
            )
            label = np.asarray(label, dtype=np.uint8)  # warning wh swapping
            label = np.uint8(label != 0)
            self.nbbat += np.sum((label == 1).astype(int))
            self.nbnonbat += np.sum((label == 0).astype(int))

        self.balance = self.nbnonbat / self.nbbat

    ###
    ### get the hole image
    ### for test usage -- or in internal call for extracting crops
    def getImageAndLabel(self, i):
        assert i < self.nbImages

        image = (
            PIL.Image.open(self.pathTOdata + str(i) + "_x.png").convert("RGB").copy()
        )
        image = np.asarray(image, dtype=np.uint8)  # warning wh swapping

        label = PIL.Image.open(self.pathTOdata + str(i) + "_y.png").convert("L").copy()
        label = np.asarray(label, dtype=np.uint8)  # warning wh swapping
        label = np.uint8(label != 0)
        return image, label

    ###
    ### get randomcrops + symetrie
    ### get train usage
    def getrawrandomtiles(self, nbtiles, tilesize):
        XY = []
        nbtilesperimage = int(nbtiles / self.nbImages + 1)

        # crop
        for name in range(self.nbImages):
            # nbtilesperimage*probaOnImage==nbtiles
            if (
                nbtiles < self.nbImages
                and random.randint(0, int(nbtiles + 1)) > nbtilesperimage
            ):
                continue

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


class MiniWorld:
    def __init__(self, flag="train", custom=None, without=None):
        assert flag in ["train", "test", "custom"]

        self.root, self.towns = getindexeddata()
        if flag == "custom":
            self.towns = custom
        else:
            if without is not None:
                self.towns = [town for town in self.towns if town not in without]
            self.towns = [town + "/" + flag for town in self.towns]

        self.data = {}
        self.nbImages = 0
        self.nbbat, self.nbnonbat = 0, 0
        for town in self.towns:
            self.data[town] = SegSemDataset(self.root + town + "/")
            self.nbImages += self.data[town].nbImages
            self.nbbat += self.data[town].nbbat
            self.nbnonbat += self.data[town].nbnonbat

        self.balance = self.nbnonbat / self.nbbat
        print(
            "indexing miniworld (mode",
            flag,
            "):",
            len(self.towns),
            "towns found (",
            self.towns,
            ") with a total of",
            self.nbImages,
            "images",
        )

    def getrandomtiles(self, nbtiles, tilesize, batchsize, mode="PerTown"):
        assert mode in ["PerImage", "PerTown", "PerPixel"]

        XY = []
        if mode == "PerImage":
            nbtilesperimage = 1.0 * nbtiles / self.nbImages

            for town in self.towns:
                XY += self.data[town].getrawrandomtiles(
                    nbtilesperimage * self.data[town].nbImages, tilesize
                )

        if mode == "PerTown":
            nbtilesperTown = 1.0 * nbtiles / len(self.towns)

            for town in self.towns:
                XY += self.data[town].getrawrandomtiles(nbtilesperTown, tilesize)

        if mode == "PerPixel":
            nbtilesperPixel = 1.0 * nbtiles / (self.nbnonbat + self.nbbat)

            for town in self.towns:
                nbpixelintown = self.data[town].nbnonbat + self.data[town].nbbat
                XY += self.data[town].getrawrandomtiles(
                    nbpixelintown * nbtilesperPixel, tilesize
                )

        # pytorch
        X = torch.stack(
            [torch.Tensor(np.transpose(x, axes=(2, 0, 1))).cpu() for x, y in XY]
        )
        Y = torch.stack([torch.from_numpy(y).long().cpu() for x, y in XY])
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchsize, shuffle=True, num_workers=2
        )

        return dataloader


def largeforward(net, image, device, tilesize=128, stride=64):
    ## assume GPU is large enough -- use largeforwardCPU on very large image
    net.eval()
    with torch.no_grad():
        pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).to(device)
        image = image.float().to(device)
        i = 0
        for row in range(0, image.shape[2] - tilesize + 1, stride):
            for col in range(0, image.shape[3] - tilesize + 1, stride):
                tmp = net(image[:, :, row : row + tilesize, col : col + tilesize])
                pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]

                # if i % 500 == 499:
                #    print("forward in progress", row, image.shape[2])
                i += 1

    return pred


def largeforwardCPU(net, image, device, tilesize=128, stride=32):
    pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).cpu()

    net.eval()
    with torch.no_grad():
        i = 0
        for row in range(0, image.shape[2] - tilesize + 1, stride):
            for col in range(0, image.shape[3] - tilesize + 1, stride):
                tmp = net(
                    image[:, :, row : row + tilesize, col : col + tilesize]
                    .float()
                    .to(device)
                ).cpu()
                pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]

                if i % 500 == 499:
                    print("forward in progress", row, image.shape[2])
                i += 1

    return pred


def convertIn3class(y):
    yy = 1.0 - y  # inverse background and building
    yy = torch.nn.functional.max_pool2d(
        yy, kernel_size=5, stride=1, padding=2
    )  # expand background
    yy = 1.0 - yy  # restore 0 - 1
    yy = yy.long()
    yy += 2 * (yy != y).long()  # work because we have extended only the background
    return yy


def convertIn3classNP(y):
    yy = convertIn3class(torch.Tensor(y).cuda().unsqueeze(0))
    return np.uint8(yy[0].cpu().numpy())
