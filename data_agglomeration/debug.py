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
    sys.path.append("/home/achanhon/github/segmentation_models/albumentations")
    sys.path.append("/home/achanhon/github/segmentation_models/imgaug")
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

import albumentations as A

transform = A.Compose(
    [
        A.OneOf(
            [
                A.ChannelDropout(),
                A.ChannelShuffle(),
                A.ColorJitter(),
                A.GaussNoise(),
                A.GaussianBlur(),
                A.HueSaturationValue(),
                A.RandomBrightnessContrast(),
                A.RandomFog(),
                A.RandomGamma(),
                A.RandomRain(),
                A.RandomSnow(),
                A.RandomSunFlare(),
            ]
        )
    ]
)


def augmentbatch(inputs):
    tmp = inputs.cpu().data.numpy()
    tmp = np.transpose(tmp, axes=(0, 2, 3, 1))
    l = []
    for i in range(tmp.shape[0]):
        l.append(transform(image=tmp[i])["image"])
    tmp = np.stack(l)
    tmp = np.transpose(tmp, axes=(0, 3, 1, 2))
    return torch.Tensor(tmp).to(device)


for epoch in ["PerImage", "PerTown", "PerPixel"]:
    print("epoch=", epoch)

    XY = miniworld.getrandomtiles(2000, 128, batchsize, mode=epoch)
    for x, y in XY:
        x, y = x.to(device), y.to(device)

        if random.randint(0, 10) == 0:
            x = augmentbatch(x)

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

# test after debug without augmentation
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
#
# test after 400 epoch of naive training
# potsdam/test 87.40174729472406
# austin/test 83.68989319229452
# chicago/test 79.11536825304486
# kitsap/test 79.61154186157935
# tyrol-w/test 86.24986425135876
# vienna/test 82.82616916237626
# christchurch/test 79.35125759839107
# vegas/test 80.9132266781993
# paris/test 75.41170459774713
# shanghai/test 72.76385834440983
# khartoum/test 70.06249857404349
# toulouse/test 83.93509259795016
# bruges/test 81.18311937839039
# rio/test 73.78848362293215
# Arlington/test 79.12604796796012
# Austin/test 85.19335374644257
# DC/test 75.87286061465227
# NewYork/test 80.60562818214026
# SanFrancisco/test 78.63391935221355
# Atlanta/test 82.73527932112401
# NewHaven/test 75.0707352818674
# Norfolk/test 82.17366323770783
# Seekonk/test 81.41157294893094
# miniworld 79.68348226834726
#
# apres acceleration du test
# potsdam/test 87.20429982300551
# austin/test 83.343100341497
# chicago/test 78.61021983508546
# kitsap/test 79.18024666076215
# tyrol-w/test 86.0398719718234
# vienna/test 82.52041133023309
# christchurch/test 79.12663519284511
# vegas/test 80.9132280546888
# paris/test 75.41170459774713
# shanghai/test 72.76385997819531
# khartoum/test 70.06251023112299
# toulouse/test 83.4980831897475
# bruges/test 80.45425459957849
# rio/test 73.64193214372764
# Arlington/test 78.83638744167843
# Austin/test 84.67683477212191
# DC/test 75.64845190639973
# NewYork/test 80.14817310233224
# SanFrancisco/test 78.35111517097585
# Atlanta/test 82.58364105043978
# NewHaven/test 74.92984211220065
# Norfolk/test 81.95843821160811
# Seekonk/test 81.02417499929513
# miniworld 79.48012397186284
