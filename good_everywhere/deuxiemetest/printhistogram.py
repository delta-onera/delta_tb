import os
import sys
import numpy
import PIL
from PIL import Image

import dataloader

miniworld = dataloader.MiniWorld(flag="train")


def fillhistogram(histogram, image):
    for i in range(255):
        histogram[i] += numpy.sum(numpy.int32(image == i))
    return histogram


chn = ["rouge", "vert", "bleu"]
for ch in range(3):
    print(chn[ch])
    for city in miniworld.cities:
        histogram = numpy.zeros(255)

        for i in range(miniworld.data[city].NB):
            im, _ = miniworld.data[city].getImageAndLabel(i)
            histogram = fillhistogram(histogram, im[:, :, ch])

        histogram = numpy.float32(histogram) / numpy.sum(histogram)
        histogram = numpy.int32(histogram * 1000.0)

        s = city
        for i in range(255):
            s = s + "\t" + str(histogram[i])
        print(s)
