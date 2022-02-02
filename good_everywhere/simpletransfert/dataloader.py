import os
import sys
import numpy
import PIL
from PIL import Image, ImageDraw
import rasterio
import torch
import json


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def torchTOpil(x):
    return numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))


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
        return torch.Tensor((iou0 + iou1, accu))
    else:
        out = torch.zeros(cm.shape[0], 2)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        return out


def minmax(x, xmin, xmax):
    out = 255 * (x - xmin) / (xmax - xmin)

    tmp = numpy.int16(out >= 255)  # the large ones
    out -= 10000 * tmp  # makes the larger small
    out *= numpy.int16(out > 0)  # put small at 0
    out += 255 * tmp  # restore large at 255

    return numpy.int16(out)


class MinMax:
    def __init__(self, data):
        print("MinMax")
        self.imin = numpy.zeros(3)
        self.imax = numpy.zeros(3)

        values = {}
        for ch in range(3):
            values[ch] = [0]
        for i in range(data.NB):
            x, _ = data.getImageAndLabel(i)
            for ch in range(3):
                values[ch] += list(x[:, :, ch].flatten())

        for ch in range(3):
            values[ch] = sorted(values[ch])
            I = len(values[ch])
            values[ch] = values[ch][(I * 3) // 100 : (I * 97) // 100]
            self.imin[ch] = values[ch][0]
            self.imax[ch] = values[ch][-1]

    def normalize(self, image):
        out = numpy.zeros(image.shape)
        for ch in range(3):
            out[:, :, ch] = minmax(image[:, :, ch], self.imin[ch], self.imax[ch])
        return numpy.int16(out)


class HistogramBased:
    def __init__(self, data, mode="flat"):
        print("HistogramBased", mode)
        self.XP = numpy.zeros((3, 256))
        self.FP = numpy.arange(256)

        target = numpy.ones(256)
        if mode == "center":
            target[256 // 4 : (3 * 256) // 4] = 10
        if mode == "left":
            target[0 : 256 // 2] = 10
        if mode == "right":
            target[256 // 2 :] = 10
        quantiles = numpy.cumsum(target)
        quantiles = quantiles / quantiles[-1]

        values = {}
        for ch in range(3):
            values[ch] = [0]
        for i in range(data.NB):
            x, _ = data.getImageAndLabel(i)
            for ch in range(3):
                values[ch] += list(x[:, :, ch].flatten())

        for ch in range(3):
            tmp = numpy.asarray(values[ch])
            _, src_indices, src_counts = numpy.unique(
                tmp, return_inverse=True, return_counts=True
            )

            # ensure single value can not distord the histogram
            cut = numpy.ones(src_counts.shape) * tmp.shape[0] / 20
            src_counts = numpy.minimum(src_counts, cut)
            src_quantiles = numpy.cumsum(src_counts)
            src_quantiles = src_quantiles / src_quantiles[-1]

            interp_a_values = numpy.interp(src_quantiles, quantiles, self.FP)
            wtf = interp_a_values[src_indices]

            for i in range(256):
                tmpi = numpy.int16(wtf < i + 1)
                last = numpy.amax(tmpi * tmp)
                self.XP[ch][i] = last + 1
            print(self.XP[ch])

    def normalize(self, image):
        out = numpy.zeros(image.shape)
        for ch in range(3):
            out[:, :, ch] = numpy.interp(image[:, :, ch], self.XP[ch], self.FP)
        return numpy.int16(out)


class Toulouse:
    def __init__(self, path="/scratchf/SEMCITY_TOULOUSE/"):
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
        y = numpy.uint8((y[:, :, 0] == 238) * (y[:, :, 1] == 118) * (y[:, :, 2] == 33))

        return x, y


def spacenet2name():
    tmp = ["2_Vegas", "4_Shanghai", "3_Paris", "5_Khartoum"]
    return ["AOI_" + name + "_Train" for name in tmp]


class SPACENET2:
    def __init__(self, name, path="/scratchf/DATASETS/SPACENET2/train/"):
        print("SPACENET2", name)
        self.path = path
        self.name = name
        assert name in spacenet2name()

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
        else:
            print(self.NB)

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

        y = numpy.uint8(numpy.asarray(mask)[:, :, 0] != 0)
        return x, y


class PhysicalData:
    def __init__(self, names=None, flag="minmax"):
        self.flag = flag

        if names is not None:
            self.cities = names
        else:
            self.cities = ["toulouse"]  # + spacenet2name()

        self.data = {}
        self.NB = {}
        self.normalization = {}
        for name in self.cities:
            if name == "toulouse":
                self.data["toulouse"] = Toulouse()
            if name in spacenet2name():
                self.data[name] = SPACENET2(name)
            self.NB[name] = self.data[name].NB

            if self.flag != "minmax":
                self.normalization[name] = HistogramBased(self.data[name], mode=flag)
            else:
                self.normalization[name] = MinMax(self.data[name])

    def getImageAndLabel(self, city, i, torchformat=False):
        x, y = self.data[city].getImageAndLabel(i)

        x = self.normalization[city].normalize(x)

        if torchformat:
            return pilTOtorch(x), torch.Tensor(y)
        else:
            return x, y
