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

    def getImageAndLabel(self, i, torchformat=False):
        x, y, name = self.paths[i]
        with rasterio.open(x) as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        y = PIL.Image.open(y).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.clip(numpy.nan_to_num(y) - 1, 0, 12)

        x, y = torch.Tensor(x), torch.Tensor(y)
        h, w = 512, 512
        x = [x]
        for mode in ["RGB", "RIE", "IGE", "IEB"]:
            path = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/APPROCHE_COMBINEE/PREPAREFUSION/build/"
            tmp = path + mode + "/train/" + name
            tmp = torch.load(tmp, map_location=torch.device("cpu"))
            tmp = tmp.unsqueeze(0).float()
            tmp = torch.nn.functional.interpolate(tmp, size=(h, w), mode="bilinear")
            x.append(tmp[0])

        x = torch.cat(x, dim=0)
        assert x.shape == (57, 512, 512)
        return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        x = torch.zeros(batchsize, 57, 512, 512)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)

        while True:
            i = int(torch.rand(1) * len(self.paths))
            x, y = self.getImageAndLabel(i)
            self.q.put((x, y), block=True)


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["1/4", "3/4"]
        self.root = root
        self.flag = flag
        self.run = False

        # TODO indiquer la sous distribution en utilisant les metadata
        # pourrait aussi faciliter l'indexation...
        self.paths = []
        level1 = os.listdir(root)
        for folder in level1:
            level2 = os.listdir(root + folder)

            for subfolder in level2:
                path = root + folder + "/" + subfolder
                level3 = os.listdir(path + "/img")
                level3 = set([name[4:] for name in level3 if ".aux" not in name])

                level3bis = os.listdir(path + "/msk")
                level3bis = [name[4:] for name in level3bis if ".aux" not in name]

                names = [name for name in level3bis if name in level3]

                for name in names:
                    x = path + "/img/IMG_" + name
                    y = path + "/msk/MSK_" + name
                    self.paths.append(("TODO", x, y, "PRED_" + name))

        # séparer les sous distributions
        self.paths = sorted(self.paths)
        tmp = [i for i in range(len(self.paths)) if i % 2 == 1]
        self.paths = [self.paths[i] for i in tmp]
        if flag == "1/4":
            tmp = [i for i in range(len(self.paths)) if i % 2 == 0]
        else:
            tmp = [i for i in range(len(self.paths)) if i % 2 == 1]
        self.paths = [self.paths[i] for i in tmp]

        self.pathssubdistrib = {}
        for i in range(len(self.paths)):
            sousdis, x, y, meta = self.paths[i]
            if sousdis not in self.pathssubdistrib:
                self.pathssubdistrib[sousdis] = []
            self.pathssubdistrib[sousdis].append((x, y, meta))
            self.paths[i] = (sousdis, len(self.pathssubdistrib[sousdis]) - 1)

        self.subdistrib = list(self.pathssubdistrib.keys())
        self.data = {}
        for sousdis in self.pathssubdistrib.keys():
            self.data[sousdis] = CropExtractor(self.pathssubdistrib[sousdis])

    def getImageAndLabel(self, i, torchformat=False):
        sousdistrib, j = self.paths[i]
        return self.data[sousdistrib].getImageAndLabel(j, torchformat=torchformat)

    def getBatch(self, batchsize):
        assert self.run
        seed = (torch.rand(batchsize) * len(self.data)).long()
        seed = [self.subdistrib[i] for i in seed]
        x = torch.zeros(batchsize, 57, 512, 512)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], y[i] = self.data[seed[i]].getCrop()
        return x, y

    def start(self):
        if not self.run:
            self.run = True
            for sousdis in self.subdistrib:
                self.data[sousdis].start()


class FusionNet(torch.nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.f1 = torch.nn.Conv2d(57, 57, kernel_size=17, padding=8)
        self.f2 = torch.nn.Conv2d(171, 57, kernel_size=9, padding=4)
        self.f3 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
        self.f4 = torch.nn.Conv2d(171, 256, kernel_size=1)
        self.f5 = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        z = x / 125
        z = torch.nn.functional.leaky_relu(self.f1(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f2(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f3(z))
        x = torch.cat([x, z, x * z], dim=1)
        x = torch.nn.functional.leaky_relu(self.f4(x))
        return self.f5(x)
