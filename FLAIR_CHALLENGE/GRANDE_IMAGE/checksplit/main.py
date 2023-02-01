import torch
import torchvision
import numpy
import rasterio
import os


def rankfromlist(l):
    l = list(set(l))
    out = {}
    for i in range(len(l)):
        out[l[i]] = i * 512
    return out


def boxesTOcoords(path, allboxes):
    boxes = numpy.load(path, allow_pickle=True)

    cols = [boxes[i][1].left for i in range(boxes.shape[0])]
    rows = [boxes[i][1].top for i in range(boxes.shape[0])]
    cols, rows = rankfromlist(cols), rankfromlist(rows)

    for i in range(boxes.shape[0]):
        image, left, top = boxes[i][0], boxes[i][1].left, boxes[i][1].top
        allboxes[image] = path, rows[top], cols[left]


allboxes = {}
domaines = os.listdir("/scratchf/flair_merged/test/")
for domaine in domaines:
    paths = os.listdir("/scratchf/flair_merged/test/" + domaine)
    paths = [path for path in paths if ".npy" in path]
    for path in paths:
        tmp = "/scratchf/flair_merged/test/" + domaine + "/" + path
        boxesTOcoords(tmp, allboxes)

print(allboxes)

quit()


path = "/scratchf/flair_merged/test/D012_2019/IMG_Z10_UU.npy"
a = numpy.load(path, allow_pickle=True)
print(a[3][0])
print(a[3][1])
print(a[3][1].top)


path = "/scratchf/CHALLENGE_IGN/test/D012_2019/Z10_UU/img/IMG_062700.tif"
with rasterio.open(path) as src_img:
    print(src_img.bounds)


a = "/scratchf/flair_merged/test/D012_2019/"
