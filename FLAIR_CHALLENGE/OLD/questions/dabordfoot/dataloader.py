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
        self.K = 3

    def getImageAndLabel(self, i, torchformat=False):
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            x = x[0:3, :, :]  # pour le moment
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        y = PIL.Image.open(self.paths[i][1]).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.clip(numpy.nan_to_num(y) - 1, 0, 12)

        # self.path[i][2] contient metadata à ajouter à x ?

        if torchformat:
            return torch.Tensor(x), torch.Tensor(y)
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        x = torch.zeros(batchsize, self.K, 512, 512)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)

        while True:
            i = int(torch.rand(1) * len(self.paths))
            flag = numpy.random.randint(0, 2, size=3)
            x, y = self.getImageAndLabel(i)
            x, y = symetrie(x, y, flag)
            x, y = torch.Tensor(x), torch.Tensor(y)
            self.q.put((x, y), block=True)


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["odd", "even", "all"]
        self.root = root
        self.flag = flag
        self.K = 3
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
                    meta = None
                    self.paths.append(("TODO", x, y, meta))

        # séparer les sous distributions
        self.paths = sorted(self.paths)
        if flag != "all":
            if flag == "even":
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
        x = torch.zeros(batchsize, self.K, 512, 512)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], y[i] = self.data[seed[i]].getCrop()
        return x, y

    def start(self):
        if not self.run:
            self.run = True
            for sousdis in self.subdistrib:
                self.data[sousdis].start()


class Mobilenet(torch.nn.Module):
    def __init__(self):
        super(Mobilenet, self).__init__()
        self.backend = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights="DEFAULT"
        )
        self.backend.classifier.low_classifier = torch.nn.Conv2d(40, 13, kernel_size=1)
        self.backend.classifier.high_classifier = torch.nn.Conv2d(
            128, 13, kernel_size=1
        )

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25
        return self.backend(x)["out"]
