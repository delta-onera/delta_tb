import os
import PIL
from PIL import Image
import numpy
import torch
import random


def distancetransform(y, size=4):
    yy = 2.0 * y.unsqueeze(0) - 1
    yyy = torch.nn.functional.avg_pool2d(
        yy, kernel_size=2 * size + 1, stride=1, padding=size
    )
    D = 1.0 - 0.5 * (yy - yyy).abs()
    return D[0]


def perf(cm):
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return torch.Tensor((iou0 + iou1, accu))


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


class CropExtractor:
    def __init__(self, path, tilesize=128):
        self.path = path
        self.NB = 0
        self.tilesize = tilesize
        while os.path.exists(self.path + str(self.NB) + "_x.png"):
            self.NB += 1

        if self.NB == 0:
            print("wrong path", self.path)
            quit()

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

    ###############################################################

    def getCrop(self, nbtiles, batchsize):
        X, Y = []
        nbpertile = int(nbtiles / self.NB + 1)

        for i in range(self.NB):
            image, label = self.getImageAndLabel(i, torchformat=False)

            RC = numpy.random.rand(nbpertile, 2)
            flag = numpy.random.randint(0, 2, size=(nbpertile, 3))
            for j in range(nbpertile):
                r = int(RC[j][0] * (image.shape[0] - tilesize - 2))
                c = int(RC[j][1] * (image.shape[1] - tilesize - 2))
                im = image[r : r + tilesize, c : c + tilesize, :]
                mask = label[r : r + tilesize, c : c + tilesize]
                x, y = symetrie(im.copy(), mask.copy(), flag[j])
                X.append(x)
                Y.append(Y)

        X = torch.stack([torch.Tensor(np.transpose(x, axes=(2, 0, 1))) for x in X])
        Y = torch.stack([torch.from_numpy(y).long() for y in Y])
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchsize, shuffle=True, num_workers=2
        )
        return dataloader
