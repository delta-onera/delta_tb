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
prednames = set([name for name in prednames if ".tif" in name])

with torch.no_grad():
    cm = torch.zeros((13, 13))
    subdists = dataset.data.keys()
    for subdist in subdists:
        paths = dataset.data[subdist].paths
        for x, y, _ in paths:
            tmp = torch.rand(1) * 1000
            if int(tmp) == 0:
                print(cm.flatten().sum() / 512 / 512, cm[:5, :5])

            if "msk/MSK_" not in y:
                continue
            i = y.index("msk/MSK_")
            name = "PRED" + y[(i + 7) :]
            if name not in prednames:
                print(name)
                quit()
                continue
            y = PIL.Image.open(y).convert("L").copy()
            y = numpy.asarray(y)
            y = numpy.clip(numpy.nan_to_num(y), 0, 12)

            z = PIL.Image.open("../build/" + name).convert("L").copy()
            z = numpy.asarray(z)

            cm += dataloader.confusion(y, z)

    print(cm)
    print(dataloader.perf(cm))
