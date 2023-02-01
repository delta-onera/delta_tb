import torch
import torchvision
import numpy
import rasterio

path = "/scratchf/flair_merged/test/D012_2019/IMG_Z10_UU.npy"
a = numpy.load(path, allow_pickle=True)
print(a)

path = "/scratchf/CHALLENGE_IGN/test/D012_2019/Z10_UU/img/IMG_062700.tif"
with rasterio.open(path) as src_img:
    print(src_img.coords.BoundingBox)
