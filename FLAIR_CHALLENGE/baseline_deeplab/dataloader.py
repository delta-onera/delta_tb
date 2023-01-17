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
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        y = numpy.transpose(y, axes=(1, 0))
        for u in x.shape[0]:
            x[u] = numpy.transpose(x[u], axes=(1, 0))
    if j == 1:
        y = numpy.flip(y, axis=0)
        for u in x.shape[0]:
            x[u] = numpy.flip(x[u], axis=0)
    if k == 1:
        y = numpy.flip(y, axis=1)
        for u in x.shape[0]:
            x[u] = numpy.flip(x[u], axis=1)
    return x.copy(), y.copy()


class CropExtractor(threading.Thread):
    def __init__(self, paths):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 500
        self.tilesize = 256
        self.paths = paths
        self.K = 3

    def getImageAndLabel(self, i, torchformat=False):
        # x = PIL.Image.open(self.paths[i][0]).convert("RGB").copy()
        # x = numpy.uint8(numpy.asarray(x))
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            x = x[0:3, :, :]  # pour le moment

        y = PIL.Image.open(self.paths[i][1]).convert("L").copy()
        y = numpy.uint8(numpy.asarray(y) != 0)

        # self.path[i][2] contient metadata à ajouter à x ?

        if torchformat:
            return torch.Tensor(x), torch.Tensor(y)
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, self.K, tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize

        while True:
            for i in range(len(self.paths)):
                image, label = self.getImageAndLabel(i)

                ntile = 3
                RC = numpy.random.rand(ntile, 2)
                flag = numpy.random.randint(0, 2, size=(ntile, 3))
                for j in range(ntile):
                    r = int(RC[j][0] * (image.shape[0] - tilesize - 2))
                    c = int(RC[j][1] * (image.shape[1] - tilesize - 2))
                    im = image[:, r : r + tilesize, c : c + tilesize]
                    mask = label[r : r + tilesize, c : c + tilesize]
                    x, y = symetrie(im.copy(), mask.copy(), flag[j])
                    x, y = torch.Tensor(x), torch.Tensor(y)
                    self.q.put((x, y), block=True)


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["odd", "even", "all"]
        self.root = root
        self.flag = flag
        self.K = 3
        self.run = False
        self.tilesize = 256

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
        x = torch.zeros(batchsize, self.K, self.tilesize, self.tilesize)
        y = torch.zeros(batchsize, self.tilesize, self.tilesize)
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


class Deeplab(torch.nn.Module):
    def __init__(self):
        super(Deeplab, self).__init__()
        self.backend = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DEFAULT"
        )
        self.backend.classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25
        return self.backend(x)["out"]


class GlobalLocal(torch.nn.Module):
    def __init__(self):
        super(GlobalLocal, self).__init__()
        self.backbone = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        self.compress = torch.nn.Conv2d(1280, 32, kernel_size=1)
        self.classiflow = torch.nn.Conv2d(1280, 13, kernel_size=1)

        self.local1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.local2 = torch.nn.Conv2d(99, 32, kernel_size=5, padding=2)
        self.local3 = torch.nn.Conv2d(99, 32, kernel_size=5, padding=2)
        self.local4 = torch.nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.local5 = torch.nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.classifhigh = torch.nn.Conv2d(32, 13, kernel_size=1)

    def forwardglobal(self, x):
        x = 2 * (x / 255) - 1
        x = torch.nn.functional.interpolate(
            x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="bilinear"
        )
        return torch.nn.functional.leaky_relu(self.backbone(x))

    def forwardlocal(self, x, f):
        z = self.local1(x)
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local2(z))
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local3(z))

        zz = torch.nn.functional.max_pool2d(z, kernel_size=3, stride=1, padding=1)
        zz = torch.nn.functional.relu(100 * z - 99 * zz)

        z = torch.cat([z, zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local4(z))
        z = torch.cat([z, z * zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local5(z))
        return self.classifhigh(z)

    def forward(self, x, mode="normal"):
        assert mode in ["normal", "globalonly", "nofinetuning"]

        if mode != "normal":
            with torch.no_grad():
                f = self.forwardglobal(x)
        else:
            f = self.forwardglobal(x)

        z = self.classiflow(f)
        z = torch.nn.functional.interpolate(
            z, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        if mode == "globalonly":
            return z

        f = torch.nn.functional.leaky_relu(self.compress(f))
        f = torch.nn.functional.interpolate(
            f, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        return self.forwardlocal(x, f) + z
