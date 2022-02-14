import os
import numpy
import PIL
from PIL import Image, ImageDraw
import json
import csv
import random
import rasterio

root = "/scratchf/"
rootminiworld = "/scratchf/miniworld3/"

if os.path.exists(rootminiworld):
    os.system("rm -rf " + rootminiworld)
    os.makedirs(rootminiworld)

TARGET_RESOLUTION = 50.0
TODO = {}
TODO["toulouse"] = root + "SEMCITY_TOULOUSE/"
TODO["spacenet2"] = root + "/DATASETS/SPACENET2/train/"


def makepath(name):
    os.makedirs(rootminiworld + name)
    os.makedirs(rootminiworld + name + "/train")
    os.makedirs(rootminiworld + name + "/test")


def minmax(x, xmin, xmax):
    out = 255 * (x - xmin) / (xmax - xmin)

    tmp = numpy.int16(out >= 255)  # the large ones
    out -= 10000 * tmp  # makes the larger small
    out *= numpy.int16(out > 0)  # put small at 0
    out += 255 * tmp  # restore large at 255

    return numpy.int16(out)


class MinMax:
    def __init__(self, maxlength=200000000):
        self.values = [[0], [0], [0]]
        self.maxlength = maxlength

    def add(self, image):
        assert self.values is not None
        for ch in range(3):
            self.values[ch] += list(image[:, :, ch].flatten())

        if len(self.values[0]) > self.maxlength:
            for ch in range(3):
                random.shuffle(self.values[ch])
                self.values[ch] = self.values[ch][0 : self.maxlength // 2]

    def froze(self):
        self.imin = numpy.zeros(3)
        self.imax = numpy.zeros(3)
        for ch in range(3):
            tmp = sorted(self.values[ch])
            I = len(tmp)
            tmp = tmp[(I * 5) // 100 : (I * 95) // 100]
            self.imin[ch] = tmp[0]
            self.imax[ch] = tmp[-1]
        del self.values
        self.values = None

    def normalize(self, image):
        out = numpy.zeros(image.shape)
        for ch in range(3):
            out[:, :, ch] = minmax(image[:, :, ch], self.imin[ch], self.imax[ch])
        return numpy.int16(out)


def resize(image=None, label=None, resolution=50.0):
    if resolution == TARGET_RESOLUTION:
        return image, label
    coef = resolution / TARGET_RESOLUTION
    size = (int(image.size[0] * coef), int(image.size[1] * coef))

    if image is not None:
        image = image.resize(size, PIL.Image.BILINEAR)
    if label is not None:
        label = image.resize(size, PIL.Image.NEAREST)
    return image, label


def resizenumpy(image=None, label=None, resolution=50.0):
    if image is not None:
        image = PIL.Image.fromarray(numpy.uint8(image))
    if label is not None:
        label = PIL.Image.fromarray(numpy.uint8(label))

    image, label = resize(image=image, label=label, resolution=resolution)
    return image, label


class Toulouse:
    def __init__(self, path):
        print("Toulouse")
        self.NB = 4
        self.path = path
        self.files = [
            ("TLS_BDSD_M_03.tif", "TLS_GT_03.tif"),
            ("TLS_BDSD_M_04.tif", "TLS_GT_04.tif"),
            ("TLS_BDSD_M_07.tif", "TLS_GT_07.tif"),
            ("TLS_BDSD_M_08.tif", "TLS_GT_08.tif"),
        ]

    def getImageAndLabel(self, i):
        assert i < self.NB

        with rasterio.open(self.path + self.files[i][0]) as src:
            r = numpy.int16(src.read(4))
            g = numpy.int16(src.read(3))
            b = numpy.int16(src.read(2))
            x = numpy.stack([r, g, b], axis=-1)

        y = PIL.Image.open(self.path + self.files[i][1]).convert("RGB").copy()
        y = numpy.asarray(y)
        y = (y[:, :, 0] == 238) * (y[:, :, 1] == 118) * (y[:, :, 2] == 33) * 255

        return x, numpy.uint8(y)


if "toulouse" in TODO:
    print("export toulouse")
    makepath("toulouse")

    data = Toulouse(TODO["toulouse"])
    normalize = MinMax()
    for i in range(data.NB):
        x, _ = data.getImageAndLabel(i)
        normalize.add(x)

    normalize.froze()

    train, test = 0, 0
    for i in range(data.NB):
        x, y = data.getImageAndLabel(i)

        x = normalize.normalize(x)
        x, y = numpy.uint8(x), numpy.uint8(y)

        x, y = resizenumpy(image=x, label=y, resolution=50)
        if i % 2 == 0:
            x.save(rootminiworld + "/toulouse/train/" + str(train) + "_x.png")
            y.save(rootminiworld + "/toulouse/train/" + str(train) + "_y.png")
            train += 1
        else:
            x.save(rootminiworld + "/toulouse/test/" + str(test) + "_x.png")
            y.save(rootminiworld + "/toulouse/test/" + str(test) + "_y.png")
            test += 1


def spacenet2name():
    tmp = ["2_Vegas", "4_Shanghai", "3_Paris", "5_Khartoum"]
    tmp = ["AOI_" + name + "_Train" for name in tmp]
    tmpbis = ["vegas", "shanghai", "paris", "khartoum"]
    return [(tmp[i], tmpbis[i]) for i in range(4)]


class SPACENET2:
    def __init__(self, name, path):
        print("SPACENET2", name)
        self.path = path
        self.name = name
        assert name in [name[0] for name in spacenet2name()]

        self.NB = 0
        tmp = os.listdir(self.path + self.name + "/RGB-PanSharpen")
        tmp = [name[:-4] for name in tmp if name[-4:] == ".tif"]
        tmp = [name[15:] for name in tmp]
        tmp2 = os.listdir(self.path + self.name + "/geojson/buildings")
        tmp = [name for name in tmp if "buildings_" + name + ".geojson" in tmp2]

        self.files = sorted(tmp)
        self.NB = len(self.files)
        if self.NB == 0:
            print("wrong path")
            quit()

    def getImageAndLabel(self, i):
        assert i < self.NB
        x = "/RGB-PanSharpen/RGB-PanSharpen_" + self.files[i] + ".tif"
        y = "/geojson/buildings/buildings_" + self.files[i] + ".geojson"

        with rasterio.open(self.path + self.name + x) as src:
            affine = src.transform
            r = numpy.int16(src.read(1))
            g = numpy.int16(src.read(2))
            b = numpy.int16(src.read(3))
        x = numpy.stack([r, g, b], axis=2)

        mask = Image.new("RGB", (r.shape[1], r.shape[0]))
        draw = ImageDraw.Draw(mask)
        with open(self.path + self.name + y, "r") as infile:
            text = json.load(infile)
        shapes = text["features"]

        for shape in shapes:
            polygonXYZ = shape["geometry"]["coordinates"][0]
            if type(polygonXYZ) != type([]):
                continue
            if len(polygonXYZ) < 3:
                continue
            polygon = [
                rasterio.transform.rowcol(affine, xyz[0], xyz[1]) for xyz in polygonXYZ
            ]
            polygon = [(y, x) for x, y in polygon]
            draw.polygon(polygon, fill="#ffffff", outline="#ffffff")

        y = numpy.asarray(mask)[:, :, 0] != 0
        return x, numpy.uint8(y * 255)


if "spacenet2" in TODO:
    print("export spacenet2")

    for name, namesimple in spacenet2name():
        makepath(namesimple)

        data = SPACENET2(name, TODO["spacenet2"])
        normalize = MinMax()
        for i in range(data.NB):
            x, _ = data.getImageAndLabel(i)
            normalize.add(x)

        normalize.froze()

        train, test = 0, 0
        for i in range(data.NB):
            x, y = data.getImageAndLabel(i)

            x = normalize.normalize(x)
            x, y = numpy.uint8(x), numpy.uint8(y)

            x, y = resizenumpy(image=x, label=y, resolution=30)
            if i % 2 == 0:
                x.save(rootminiworld + namesimple + "/train/" + str(train) + "_x.png")
                y.save(rootminiworld + namesimple + "/train/" + str(train) + "_y.png")
                train += 1
            else:
                x.save(rootminiworld + namesimple + "/test/" + str(test) + "_x.png")
                y.save(rootminiworld + namesimple + "/test/" + str(test) + "_y.png")
                test += 1
