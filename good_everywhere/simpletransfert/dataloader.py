import os
import sys
import numpy
import PIL
from PIL import Image
import torch
import random
import normalization


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
        self.cible[1][0 : 256 // 2] = 10
        self.cible[2][256 // 4 : (3 * 256) // 4] = 10
        self.cible[3][256 // 2 :] = 10

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

    def __call__(self, image, flag=None):
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


class PhysicalData:
    def __init__(self, path, name, flag):
        self.path = path
        self.name = name
        self.flag = flag

        if self.name == "semcity_toulouse":
            self.files = [
                ("TLS_BDSD_M_03.tif", "TLS_GT_03.tif"),
                ("TLS_BDSD_M_03.tif", "TLS_GT_04.tif"),
                ("TLS_BDSD_M_03.tif", "TLS_GT_07.tif"),
                ("TLS_BDSD_M_03.tif", "TLS_GT_08.tif"),
            ]
