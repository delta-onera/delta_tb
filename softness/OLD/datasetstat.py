import os
import PIL
from PIL import Image
import numpy


def getVoisinage(y):
    l = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)]

    h, w = y.shape[0], y.shape[1]
    out = numpy.zeros((9, h + 2, w + 2))
    for i, (dh, dw) in enumerate(l):
        out[i, 1 + dh : 1 + dh + h, 1 + dw : 1 + dw + w] = y[:, :]

    return labelbis[:, 1 : 1 + h, 1 : 1 + w]


def isborder(y):
    voisinage = getVoisinage(y)
    voisinage = numpy.sum(voisinage, axis=0)
    return numpy.uint32(y * 9 != voisinage)


def getstat(path, C=2):
    names = os.listdir(path)
    names = [name for name in names if "_y.png" in name]

    stats = numpy.zeros(C + 1)
    for name in names:
        y = PIL.Image.open(path + "/" + name).convert("L").copy()
        y = numpy.uint8(numpy.asarray(y))
        if C == 2:
            y = numpy.uint8(y != 0)

        for c in range(C):
            stats[y] += numpy.sum(numpy.int32(y == c))
        stats[-1] = numpy.sum(isborder(y))

    return stats


print("/scratchf/miniworld/christchurch/")
stats = getstat("/scratchf/miniworld/christchurch/")
print(stats, stats / stats.shape[0] / stats.shape[1])

print("/scratchf/miniworld_1M/christchurch/")
stats = getstat("/scratchf/miniworld/christchurch/")
print(stats, stats / stats.shape[0] / stats.shape[1])
