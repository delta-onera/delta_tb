import os
import torch
import dataloader
import PIL
from PIL import Image
import numpy

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/train/", "all")

with torch.no_grad():
    histo = torch.zeros(13)
    subdists = dataset.data.keys()
    for subdist in subdists:
        print("==>subdist")
        paths = dataset.data[subdist].paths
        for x, y, _ in paths:
            tmp = torch.rand(1) * 100
            if int(tmp) == 0:
                print(histo.flatten().sum() / 512 / 512, histo)

            y = PIL.Image.open(y).convert("L").copy()
            y = numpy.asarray(y)
            y = numpy.clip(numpy.nan_to_num(y), 0, 12)
            y = torch.Tensor(y)

            for i in range(13):
                histo[i] += (y == i).sum()

    print(histo)
