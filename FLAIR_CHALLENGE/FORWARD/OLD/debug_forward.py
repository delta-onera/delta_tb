import os
import torch
import torchvision
import dataloader
import PIL
from PIL import Image
import numpy

assert torch.cuda.is_available()

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/train/", "all")

print("load prediction")
prediction = {}
prednames = os.listdir("../build")
prednames = [name for name in prednames if ".tif" in name]
for name in prednames:
    z = PIL.Image.open("../build/" + name).convert("L").copy()
    z = numpy.asarray(z)
    prednames[name[5:]] = z


print("compare")

with torch.no_grad():
    cm = torch.zeros((13, 13)).cuda()
    subdists = dataset.data.keys()
    for subdist in subdists:
        for x, y, _ in dataset.data[subdists].paths:
            if "img/IMG_" not in y:
                continue
            i = y.index("img/IMG_")
            name = y[(i + 7) :]
            if name in y:
                y = PIL.Image.open(y).convert("L").copy()
                y = numpy.asarray(y)
                y = numpy.clip(numpy.nan_to_num(y), 0, 12)

                z = prednames[name]

                cm += dataset.confusion(y, z)

    print(cm)
    print(dataset.perf(cm))
