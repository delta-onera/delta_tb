import os
import sys
import numpy
import PIL
from PIL import Image

import dataloader

print("load data")
miniworld = dataloader.MiniWorld(flag="train")


def fillhistogram(histogram, image):
    for ch in range(3):
        for i in range(255):
            histogram[ch][i] += numpy.sum(numpy.int32(image == i))
    return histogram


print("histo")
for city in miniworld.cities:
    histogram = numpy.zeros(3, 255)

    for i in range(miniworld.data[city].NB):
        im, _ = miniworld.data[city].getImageAndLabel(i)

        histogram = fillhistogram(histogram, im)

    print(city, histogram)
