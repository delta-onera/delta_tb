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


class HandMadeNormalization:
    def __init__(self, flag="minmax"):
        self.flag = flag

        self.cibles = numpy.ones((4, 256))
        self.cibles[1][0 : 256 // 2] = 10
        self.cibles[2][256 // 4 : (3 * 256) // 4] = 10
        self.cibles[3][256 // 2 :] = 10

        self.quantiles = []
        for i in range(4):
            quantiles = numpy.cumsum(self.cibles[i])
            quantiles = quantiles / quantiles[-1]
            self.quantiles.append(quantiles)

    def minmax(self, image, removeborder=True):
        values = list(image.flatten())
        if removeborder:
            values = sorted(values)
            I = len(values)
            values = values[(I * 3) // 100 : (I * 97) // 100]
            imin = values[0]
            imax = values[-1]
        else:
            imin = min(values)
            imax = max(values)

        if imin == imax:
            return numpy.int16(256 // 2 * numpy.ones(image.shape))

        out = 255.0 * (image - imin) / (imax - imin)
        out = numpy.int16(out)

        tmp = numpy.int16(out >= 255)
        out -= 10000 * tmp
        out *= numpy.int16(out > 0)
        out += 255 * tmp
        return out

    def histogrammatching(image, tmpl_quantiles):
        # inspired from scikit-image/blob/main/skimage/exposure/histogram_matching.py
        _, src_indices, src_counts = numpy.unique(
            image.flatten(), return_inverse=True, return_counts=True
        )

        # ensure single value can not distord the histogram
        cut = numpy.ones(src_counts.shape) * image.shape[0] * image.shape[1] / 20
        src_counts = numpy.minimum(src_counts, cut)
        src_quantiles = numpy.cumsum(src_counts)
        src_quantiles = src_quantiles / src_quantiles[-1]

        interp_a_values = numpy.interp(src_quantiles, tmpl_quantiles, numpy.arange(256))
        tmp = interp_a_values[src_indices].reshape(image.shape)
        return self.minmax(tmp, removeborder=False)

    def normalize(self, image, flag=None):
        if flag is None:
            flag = self.flag

        if flag == "minmax":
            return self.minmax(image)

        if flag == "flat":
            return self.histogrammatching(image, self.quantiles[0])

        if flag == "gaussian_left":
            return self.histogrammatching(image, self.quantiles[1])
        if flag == "gaussian":
            return self.histogrammatching(image, self.quantiles[2])
        if flag == "gaussian_right":
            return self.histogrammatching(image, self.quantiles[3])

        print("bad option in HandMadeNormalization()")
        quit()


class Toulouse:
    def __init__(self, path="/scratchf/SEMCITY_TOULOUSE/"):
        self.NB = 4
        self.path = path
        self.files = [
            ("TLS_BDSD_M_03.tif", "TLS_GT_03.tif"),
            ("TLS_BDSD_M_03.tif", "TLS_GT_04.tif"),
            ("TLS_BDSD_M_03.tif", "TLS_GT_07.tif"),
            ("TLS_BDSD_M_03.tif", "TLS_GT_08.tif"),
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
        y = (y[:, :, 0] == 238) * (y[:, :, 1] == 118) * (y[:, :, 2] == 33)
        y = numpy.uint8((y != 0) * 255)

        return x, y


def spacenet2name():
    tmp = ["2_Vegas", "4_Shanghai", "3_Paris", "5_Khartoum"]
    return ["AOI_" + name + "_Train" for name in tmp]


class SPACENET2:
    def __init__(self, name, path="/scratchf/DATASETS/SPACENET2/train/"):
        self.path = path
        self.name = name
        assert name in spacenet2name()

        self.NB = 0
        tmp = os.listdir(self.path + self.name + "/RGB-PanSharpen")
        print(tmp)
        tmp = [name[:-4] for name in tmp if name[-4:] == ".tif"]
        print(tmp)
        tmp = [name[15:] for name in tmp]
        print(tmp)
        tmp2 = os.listdir(self.path + self.name + "/geojson/buildings")
        tmp = [name for name in tmp if "buildings_" + name + ".geojson" in tmp2]

        self.files = sorted(tmp)
        NB = len(self.files)
        if NB == 0:
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

        y = numpy.uint8(numpy.asarray(mask))
        return x, y


class PhysicalData(HandMadeNormalization):
    def __init__(self, names=None, flag="minmax"):
        super().__init__(flag)
        self.flag = flag

        if names is not None:
            self.cities = names
        else:
            self.cities = ["toulouse"] + spacenet2name()

        self.data = {}
        self.NB = {}
        for name in self.cities:
            if name == "toulouse":
                self.data["toulouse"] = Toulouse()
            if name in spacenet2name():
                self.data[name] = SPACENET2(name)
            self.NB[name] = self.data[name].NB

    def getImageAndLabel(self, city, i, torchformat=False):
        x, y = self.data[city].getImageAndLabel(i)

        x = self.normalize(x)

        if torchformat:
            return pilTOtorch(x), torch.Tensor(y)
        else:
            return x, y
