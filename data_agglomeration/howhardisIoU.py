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

print("load data")
import dataloader

miniworld = dataloader.MiniWorld(
    flag="custom", custom=["potsdam/test", "bruges/test", "toulouse/test"]
)


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
        cm[town] = np.zeros((2, 2), dtype=int)
        for i in range(miniworld.data[town].nbImages):
            imageraw, label = miniworld.data[town].getImageAndLabel(i)

            pred = torch.Tensor(label).cuda().unsqueeze(0).float()
            pred = torch.nn.functional.max_pool2d(
                pred, kernel_size=3, stride=1, padding=1
            )
            pred = np.uint8(pred[0].cpu().numpy())

            cm[town] += confusion_matrix(label.flatten(), pred.flatten(), labels=[0, 1])

            imageraw = PIL.Image.fromarray(np.uint8(imageraw))
            imageraw.save("build/" + "potsdam_test_" + str(i) + "_x.png")
            labelim = PIL.Image.fromarray(np.uint8(label) * 255)
            labelim.save("build/" + "potsdam_test_" + str(i) + "_y.png")
            predim = PIL.Image.fromarray(np.uint8(pred) * 255)
            predim.save("build/" + "potsdam_test_" + str(i) + "_z.png")

        print(cm[town][0][0], cm[town][0][1], cm[town][1][0], cm[town][1][1])
        print(
            accu(cm[town]),
            f1(cm[town]),
        )

print("-------- oracle results ----------")
for town in miniworld.towns:
    print(town, accu(cm[town]), f1(cm[town]))

globalcm = np.zeros((2, 2), dtype=int)
for town in miniworld.towns:
    globalcm += cm[town]
print("miniworld", accu(globalcm), f1(globalcm))
