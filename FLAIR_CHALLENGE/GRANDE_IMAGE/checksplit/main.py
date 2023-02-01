import torch
import torchvision
import numpy
import rasterio

import os


def boxesTOcoords(path):
    boxes = numpy.load(path, allow_pickle=True)


path = "/scratchf/flair_merged/test/D012_2019/IMG_Z10_UU.npy"
a = numpy.load(path, allow_pickle=True)
print(a[3][0])
print(a[3][1])
print(a[3][1].top)


path = "/scratchf/CHALLENGE_IGN/test/D012_2019/Z10_UU/img/IMG_062700.tif"
with rasterio.open(path) as src_img:
    print(src_img.bounds)


a = "/scratchf/flair_merged/test/D012_2019/"
