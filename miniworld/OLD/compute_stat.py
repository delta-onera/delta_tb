import os
import PIL
from PIL import Image
import numpy
import torch


def distancetransform(y, size=4):
    yy = 2.0 * y.unsqueeze(0) - 1
    yyy = torch.nn.functional.avg_pool2d(
        yy, kernel_size=2 * size + 1, stride=1, padding=size
    )
    D = 1.0 - 0.5 * (yy - yyy).abs()
    return D[0]


def perf(cm):
    if len(cm.shape) == 2:
        accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
        iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
        iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
        return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))
    else:
        out = torch.zeros(cm.shape[0], 4)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        return out


class City:
    def __init__(self, path):
        self.path = path
        self.NB = 0
        self.tilesize = tilesize
        while os.path.exists(self.path + str(self.NB) + "_x.png"):
            self.NB += 1

        if self.NB == 0:
            print("wrong path", self.path)
            quit()

    def getImageAndLabel(self, i):
        assert i < self.NB

        image = PIL.Image.open(self.path + str(i) + "_x.png").convert("RGB").copy()
        image = numpy.uint8(numpy.asarray(image))

        label = PIL.Image.open(self.path + str(i) + "_y.png").convert("L").copy()
        label = numpy.uint8(numpy.asarray(label))
        label = numpy.uint8(label != 0)

        return image, label


class MiniWorld:
    def __init__(self, flag, custom=None):
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
        self.infos["shanghai"] = {"size": "large", "label": "osm"}
        self.infos["vegas"] = {"size": "large", "label": "osm"}
        self.infos["khartoum"] = {"size": "large", "label": "osm"}

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
            self.data[city] = CropExtractor(self.root + city + flag)

        self.NB = len(self.cities)

    def printstat(self):
        for city in self.infos:
            nbpixel = 0
            for i in range(self.data[city].NB):
                x, y = self.data[city].getImageAndLabel(i)
                nbpixel += y.shape[0] * y.shape[1]
            print(city, nbpixel)


if __name__ == "main":
    miniworld = MiniWorld("train")
    miniworld.printstat()
