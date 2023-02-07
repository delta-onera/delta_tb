import os
import PIL
from PIL import Image
import rasterio
import numpy
import torch
import queue
import threading
import torchvision


def confusion(y, z):
    cm = torch.zeros(13, 13).cuda()
    for a in range(13):
        for b in range(13):
            cm[a][b] = ((z == a).float() * (y == b).float()).sum()
    return cm


def perf(cm):
    cmt = torch.transpose(cm, 0, 1)

    accu = 0
    for i in range(12):
        accu += cm[i][i]
    accu /= cm[0:12, 0:12].flatten().sum()

    iou = 0
    for i in range(12):
        inter = cm[i][i]
        union = cm[i].sum() + cmt[i].sum() - cm[i][i] + (cm[i][i] == 0)
        iou += inter / union

    return (iou / 12 * 100, accu * 100)


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        y = numpy.transpose(y, axes=(1, 0))
        for u in range(x.shape[0]):
            x[u] = numpy.transpose(x[u], axes=(1, 0))
    if j == 1:
        y = numpy.flip(y, axis=0)
        for u in range(x.shape[0]):
            x[u] = numpy.flip(x[u], axis=0)
    if k == 1:
        y = numpy.flip(y, axis=1)
        for u in range(x.shape[0]):
            x[u] = numpy.flip(x[u], axis=1)
    return x.copy(), y.copy()


class CropExtractor(threading.Thread):
    def __init__(self, paths):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 500
        self.paths = paths
        self.prepa = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/GRANDE_IMAGE/PREPAREFUSION/build/"

        self.minmax = {}
        for mode in ["RGB", "RIE", "IGE", "IEB"]:
            xmin, xmax = 100000, -100000
            moyennemoyenne, variancemoyenne = 0, 0
            l = os.listdir(self.prepa + mode)
            l = [name for name in l if ".tif" in name]
            for name in l:
                tmp = torch.load(self.prepa + mode + "/" + name).float()
                localmin = tmp.flatten().min()
                if localmin < xmin:
                    xmin = localmin
                localmax = tmp.flatten().max()
                if xmax < localmax:
                    xmax = localmax

                moyennemoyenne += float(tmp.mean())
                variancemoyenne += float(torch.sqrt(tmp.var()))

            moyennemoyenne, variancemoyenne = (
                moyennemoyenne / len(l),
                variancemoyenne / len(l),
            )
            self.minmax[mode] = (
                float(xmin),
                float(xmax),
                moyennemoyenne,
                variancemoyenne,
                moyennemoyenne - variancemoyenne,
                moyennemoyenne + variancemoyenne,
            )

        self.wtf = torch.zeros(100)
        for mode in ["RGB"]:
            l = os.listdir(self.prepa + mode)
            l = [name for name in l if ".tif" in name]
            for name in l:
                tmp = torch.load(self.prepa + mode + "/" + name).float()
                tmp = (tmp - self.minmax["RGB"][0]) / (
                    self.minmax["RGB"][1] - self.minmax["RGB"][0]
                )
                tmp = (tmp * 100).long()

                for lol in range(100):
                    self.wtf[lol] += (tmp == lol).float().sum()
        for lol in range(100):
            print(lol, self.wtf[lol])


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["oddeven", "oddodd"]
        self.root = root
        self.flag = flag
        self.run = False
        self.domaines = os.listdir(root)
        self.paths = []
        for domaine in self.domaines:
            names = os.listdir(root + domaine)
            backup = set(names)
            names = [name[4:] for name in names if "MSK_" in name]
            names = [name for name in names if "IMG_" + name in backup]

            for name in names:
                y = root + domaine + "/MSK_" + name
                x = root + domaine + "/IMG_" + name
                self.paths.append((x, y, domaine + "_IMG_" + name))

        self.paths = sorted(self.paths)
        tmp = [i for i in range(len(self.paths)) if i % 2 == 1]
        self.paths = [self.paths[i] for i in tmp]

        if flag == "oddeven":
            tmp = [i for i in range(len(self.paths)) if i % 2 == 0]
        else:
            tmp = [i for i in range(len(self.paths)) if i % 2 == 1]
        self.paths = [self.paths[i] for i in tmp]

        self.data = CropExtractor(self.paths)

    def getImageAndLabel(self, i):
        return self.data.getImageAndLabel(i, torchformat=True)

    def getBatch(self, batchsize):
        return self.data.getBatch(batchsize)

    def start(self):
        if not self.run:
            self.run = True
            self.data.start()


flair = FLAIR("/scratchf/flair_merged/train/", "oddodd")
print(flair.data.minmax)
