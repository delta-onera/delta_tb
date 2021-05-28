import os
import sys
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

whereIam = os.uname()[1]


print("massif benchmark")
import dataloader

if whereIam == "super":
    miniworld = dataloader.MiniWorld(
        flag="custom", custom=["potsdam/test", "bruges/test"]
    )
else:
    miniworld = dataloader.MiniWorld("test")


def accu(cm):
    return 100.0 * (cm[0][0] + cm[1][1]) / np.sum(cm)


def f1(cm):
    return 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1]) + 50.0 * cm[1][1] / (
        cm[1][1] + cm[1][0] + cm[0][1]
    )


cm = {}
with torch.no_grad():
    for town in miniworld.towns:
        print(town)
        cm[town] = np.zeros((3, 3), dtype=int)
        for i in range(miniworld.data[town].nbImages):
            imageraw, label = miniworld.data[town].getImageAndLabel(i)

            if False:
                # 100%
                pred = label.copy()
                label = dataloader.convertIn3classNP(label)

            if False:
                # 100%
                pred = dataloader.convertIn3classNP(label).astype(int)
                pred = np.abs(pred - 1)
                pred = np.uint8(np.abs(1 - pred))
                label = dataloader.convertIn3classNP(label)

            if True:
                # ONLY AROUND 90%
                pred = dataloader.convertIn3classNP(label).astype(int)
                pred = np.abs(pred - 1)
                pred = np.uint8(np.abs(1 - pred))

            assert label.shape == pred.shape

            cm[town] += confusion_matrix(
                label.flatten(), pred.flatten(), labels=[0, 1, 2]
            )

            if True:
                imageraw = PIL.Image.fromarray(np.uint8(imageraw))
                imageraw.save("build/" + town[0:-5] + "_" + str(i) + "_x.png")
                labelim = PIL.Image.fromarray(np.uint8(label) * 125)
                labelim.save("build/" + town[0:-5] + "_" + str(i) + "_y.png")
                predim = PIL.Image.fromarray(np.uint8(pred) * 125)
                predim.save("build/" + town[0:-5] + "_" + str(i) + "_z.png")

        cm[town] = cm[town][0:2, 0:2]
        print(cm[town][0][0], cm[town][0][1], cm[town][1][0], cm[town][1][1])
        print(
            accu(cm[town]),
            f1(cm[town]),
        )

print("-------- results ----------")
for town in miniworld.towns:
    print(town, accu(cm[town]), f1(cm[town]))

globalcm = np.zeros((2, 2), dtype=int)
for town in miniworld.towns:
    globalcm += cm[town]
print("miniworld", accu(globalcm), f1(globalcm))
