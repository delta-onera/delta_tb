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

print("define model")
if whereIam == "super":
    sys.path.append("/home/achanhon/github/segmentation_models/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/segmentation_models/pytorch-image-models")
    sys.path.append(
        "/home/achanhon/github/segmentation_models/pretrained-models.pytorch"
    )
    sys.path.append(
        "/home/achanhon/github/segmentation_models/segmentation_models.pytorch"
    )
if whereIam == "wdtim719z":
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append(
        "/d/achanhon/github/EfficientNet-PyTorch"
    )  # https://github.com/lukemelas/EfficientNet-PyTorch
    sys.path.append(
        "/d/achanhon/github/pytorch-image-models"
    )  # https://github.com/rwightman/pytorch-image-models
    sys.path.append(
        "/d/achanhon/github/pretrained-models.pytorch"
    )  # https://github.com/Cadene/pretrained-models.pytorch
    sys.path.append(
        "/d/achanhon/github/segmentation_models.pytorch"
    )  # https://github.com/qubvel/segmentation_models.pytorch

import segmentation_models_pytorch as smp
import collections
import random

net = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
net = net.cuda()
net.train()


print("load data")
import dataloader

miniworld = dataloader.MiniWorld(
    flag="custom", custom=["potsdam/train", "bruges/train", "toulouse/test"]
)

earlystopping = miniworld.getrandomtiles(1000, 128, 32)
weights = torch.Tensor([1, miniworld.balance]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

print("train")


def accu(cm):
    return 100.0 * (cm[0][0] + cm[1][1]) / np.sum(cm)


def trainaccuracy():
    cm = np.zeros((2, 2), dtype=int)
    net.eval()
    with torch.no_grad():
        for inputs, targets in earlystopping:
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, pred = outputs.max(1)
            for i in range(pred.shape[0]):
                cm += confusion_matrix(
                    pred[i].cpu().numpy().flatten(),
                    targets[i].cpu().numpy().flatten(),
                    labels=[0, 1],
                )
    return cm


optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
batchsize = 16

for epoch in ["PerImage", "PerTown", "PerPixel"]:
    print("epoch=", epoch)

    XY = miniworld.getrandomtiles(2000, 128, batchsize, mode=epoch)
    for x, y in XY:
        x, y = x.to(device), y.to(device)

        preds = net(x)
        loss = criterion(preds, y)
        meanloss.append(loss.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    print("backup model")
    torch.save(net, "build/model.pth")
    cm = trainaccuracy()
    print("accuracy", accu(cm))

    if accu(cm) > 98:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")

# test after debug
# potsdam/test 76.03397522580403
# austin/test 63.38183209720267
# chicago/test 61.7307326868583
# kitsap/test 65.1367514910424
# tyrol-w/test 69.7892193296274
# vienna/test 59.49396033466258
# christchurch/test 68.45079703223041
# vegas/test 66.19719213905618
# paris/test 64.39131104772663
# shanghai/test 61.75666008100488
# khartoum/test 59.21783463541323
# toulouse/test 74.10066257585707
# bruges/test 70.67426870275182
# rio/test 66.96492822713938
# Arlington/test 57.76920544937232
# Austin/test 63.09189427565662
# DC/test 64.11312851085287
# NewYork/test 29.875100532417548
# SanFrancisco/test 50.8024765477915
# Atlanta/test 63.526100818877836
# NewHaven/test 53.45669738519585
# Norfolk/test 52.80609826576918
# Seekonk/test 62.137192840943726
# miniworld 66.23998148114374
#
# test after 80 epoch of naive training
# potsdam/test 86.12068979704375
# austin/test 83.19642038934143
# chicago/test 78.36401606565853
# kitsap/test 79.74593490788696
# tyrol-w/test 84.74266440002975
# vienna/test 80.66008897971558
# christchurch/test 76.77768326155163
# vegas/test 76.40358515960456
# paris/test 73.8083910470105
# shanghai/test 70.18845170434447
# khartoum/test 69.99339127801736
# toulouse/test 80.38447360116163
# bruges/test 79.93197113906949
# rio/test 71.54331677660981
# Arlington/test 78.03637524857203
# Austin/test 84.71786186688351
# DC/test 75.59459971995065
# NewYork/test 77.16452512032369
# SanFrancisco/test 77.68086712643813
# Atlanta/test 80.43081109094442
# NewHaven/test 73.50884682942521
# Norfolk/test 80.65416990456862
# Seekonk/test 79.3562899942964
# miniworld 77.79130455198577
