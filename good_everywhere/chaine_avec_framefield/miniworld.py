import os
import PIL
from PIL import Image
import numpy
import torch
import random
import queue
import threading


class Sobel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        G = torch.stack([Gx, Gy])
        self.G = G.unsqueeze(1).float()

    def forward(self, yz):
        tmp = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        tmp.weight = torch.nn.Parameter(self.G.clone(), requires_grad=True)
        tmp.cuda()

        x = tmp(yz)
        norm = torch.sqrt(torch.sum(x * x, dim=1))
        norm = torch.stack([norm, norm], dim=1)
        x = x / (norm + 0.001)
        return x, (norm[:, 0].detach().clone() > 0.0001).int()


def perf(cm):
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=0), numpy.flip(y, axis=0)
    return x.copy(), y.copy()


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def torchTOpil(x):
    return numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))


class CropExtractor(threading.Thread):
    def __init__(self, path, maxsize=500, tilesize=128):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = maxsize

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

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, self.tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y.long()

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize

        while True:
            I = [i for i in range(self.NB)]
            random.shuffle(I)
            for i in I:
                image, label = self.getImageAndLabel(i, torchformat=False)

                ntile = image.shape[0] * image.shape[1] // 65536 + 1
                ntile = int(min(128, ntile))

                RC = numpy.random.rand(ntile, 2)
                flag = numpy.random.randint(0, 2, size=(ntile, 3))
                for j in range(ntile):
                    r = int(RC[j][0] * (image.shape[0] - tilesize - 2))
                    c = int(RC[j][1] * (image.shape[1] - tilesize - 2))
                    im = image[r : r + tilesize, c : c + tilesize, :]
                    mask = label[r : r + tilesize, c : c + tilesize]
                    x, y = symetrie(im.copy(), mask.copy(), flag[j])
                    x, y = pilTOtorch(x), torch.Tensor(y)
                    self.q.put((x, y), block=True)


class MiniWorld:
    def __init__(self, flag, tilesize=128, custom=None):
        assert flag in ["/train/", "/test/"]

        self.tilesize = tilesize
        self.root = "/scratchf/miniworld/"

        self.infos = {}
        self.infos["potsdam"] = {"size": "small", "label": "manual"}
        self.infos["bruges"] = {"size": "small", "label": "manual"}
        self.infos["Arlington"] = {"size": "small", "label": "osm"}
        self.infos["NewHaven"] = {"size": "small", "label": "osm"}
        self.infos["Norfolk"] = {"size": "small", "label": "osm"}
        self.infos["Seekonk"] = {"size": "small", "label": "osm"}
        self.infos["Atlanta"] = {"size": "small", "label": "osm"}
        self.infos["Austin"] = {"size": "small", "label": "osm"}
        self.infos["DC"] = {"size": "small", "label": "osm"}
        self.infos["NewYork"] = {"size": "small", "label": "osm"}
        self.infos["SanFrancisco"] = {"size": "small", "label": "osm"}
        self.infos["chicago"] = {"size": "medium", "label": "osm"}
        self.infos["kitsap"] = {"size": "medium", "label": "osm"}
        self.infos["austin"] = {"size": "medium", "label": "osm"}
        self.infos["tyrol-w"] = {"size": "medium", "label": "osm"}
        self.infos["vienna"] = {"size": "medium", "label": "osm"}
        self.infos["rio"] = {"size": "large", "label": "osm"}
        self.infos["christchurch"] = {"size": "large", "label": "manual"}
        self.infos["pologne"] = {"size": "large", "label": "manual"}
        # self.infos["shanghai"] = {"size": "large", "label": "osm"}
        # self.infos["vegas"] = {"size": "large", "label": "osm"}
        # self.infos["khartoum"] = {"size": "large", "label": "osm"}

        existingcities = os.listdir(self.root)
        for city in self.infos:
            if city not in existingcities:
                print("missing city", city)
                quit()
        if custom is None:
            self.cities = [name for name in self.infos]
        else:
            self.cities = custom
        print("correctly found", self.cities)

        self.data = {}
        self.run = False
        for city in self.cities:
            self.data[city] = CropExtractor(self.root + city + flag, tilesize=tilesize)

        self.NB = len(self.cities)
        self.priority = numpy.ones(self.NB)
        self.goodlabel = numpy.zeros(self.NB)
        for i, name in enumerate(self.cities):
            if self.infos[name]["label"] == "manual":
                self.priority[i] += 1
                self.goodlabel[i] = 1
            if self.infos[name]["size"] == "medium":
                self.priority[i] += 1
            if self.infos[name]["size"] == "large":
                self.priority[i] += 2
        self.priority = numpy.float32(self.priority) / numpy.sum(self.priority)

    def start(self):
        if not self.run:
            self.run = True
            for city in self.cities:
                self.data[city].start()

    def getBatch(self, batchsize):
        assert self.run
        batchchoice = numpy.random.choice(self.NB, batchsize, p=self.priority)

        x = torch.zeros(batchsize, 3, self.tilesize, self.tilesize)
        y = torch.zeros(batchsize, self.tilesize, self.tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.data[self.cities[batchchoice[i]]].getCrop()
        return x, y.long()
