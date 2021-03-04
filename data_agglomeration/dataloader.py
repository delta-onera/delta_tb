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
        nbtilesperimage = nbtiles // self.nbImages + 1

        # crop
        for name in range(self.nbImages):
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

    ###
    ### get randomcrops + symetrie
    ### get train usage with a single dataset
    def getrandomtiles(self, nbtiles, tilesize, batchsize):
        XY = self.getrawrandomtiles(nbtiles, tilesize)

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


def largeforward(net, image, device, tilesize=128, stride=32):
    pred = torch.zeros(2, image.shape[2], image.shape[3]).cpu()

    for row in range(0, image.shape[2] - tilesize + 1, stride):
        for col in range(0, image.shape[3] - tilesize + 1, stride):
            tmp = net(
                image[:, :, row : row + tilesize, col : col + tilesize]
                .float()
                .to(device)
            ).cpu()
            pred[:, row : row + tilesize, col : col + tilesize] += tmp[0]

    return pred


def getindexeddata():
    whereIam = os.uname()[1]

    if whereIam == "super":
        availabledata = ["toulouse", "potsdam"]
        root = "/data/miniworld/"

    if whereIam == "wdtim719z":
        availabledata = ["toulouse", "potsdam"]
        root = "/data/miniworld/"

    if whereIam == "ldtis706z":
        availabledata = [
            "toulouse",
            "potsdam",
            "bruges",
            "newzealand",
        ]
        root = "/media/achanhon/bigdata/data/miniworld/"

    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        availabledata = [
            "toulouse",
            "potsdam",
            "bruges",
            "newzealand",
            "Angers",
            "Caen",
            "Cherbourg",
            "Lille_Arras_Lens_Douai_Henin",
            "Marseille_Martigues",
            "Nice",
            "Rennes",
            "Vannes",
            "Brest",
            "Calais_Dunkerque",
            "Clermont-Ferrand",
            "LeMans",
            "Lorient",
            "Nantes_Saint-Nazaire",
            "Quimper",
            "Saint-Brieuc",
        ]
        root = "TODO"

    weaklysupervised = [
        "Angers",
        "Caen",
        "Cherbourg",
        "Lille_Arras_Lens_Douai_Henin",
        "Marseille_Martigues",
        "Nice",
        "Rennes",
        "Vannes",
        "Brest",
        "Calais_Dunkerque",
        "Clermont-Ferrand",
        "LeMans",
        "Lorient",
        "Nantes_Saint-Nazaire",
        "Quimper",
        "Saint-Brieuc",
    ]

    return root, availabledata, weaklysupervised
