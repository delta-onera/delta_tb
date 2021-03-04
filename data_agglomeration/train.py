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
assert whereIam in ["super", "wdtim719z", "ldtis706z"]


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
if whereIam == "ldtis706z":
    sys.path.append("/home/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/pytorch-image-models")
    sys.path.append("/home/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/home/achanhon/github/segmentation_models.pytorch")
if whereIam == "wdtim719z":
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")

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
if whereIam in ["super", "wdtim719z"]:
    availabledata = ["toulouse", "potsdam"]
    root = "/data/miniworld/"

if whereIam == "ldtis706z":
    availabledata = [
        "toulouse",
        "potsdam",
        "bruges",
        "newzealand",
    ]
    root = "/media/achanhon/bigdata/data/miniworld/"

if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    availabledata = [
        "toulouse",
        "potsdam",
        "bruges",
        "newzealand",
        "Angers",
        "Caen",
        "Cherbourg",
        "Lille_Arras_Lens_Douai_Henin",
        "Marseille_Martigues",
        "Nice",
        "Rennes",
        "Vannes",
        "Brest",
        "Calais_Dunkerque",
        "Clermont-Ferrand",
        "LeMans",
        "Lorient",
        "Nantes_Saint-Nazaire",
        "Quimper",
        "Saint-Brieuc",
    ]
    root = "TODO"

weaklysupervised = [
    "Angers",
    "Caen",
    "Cherbourg",
    "Lille_Arras_Lens_Douai_Henin",
    "Marseille_Martigues",
    "Nice",
    "Rennes",
    "Vannes",
    "Brest",
    "Calais_Dunkerque",
    "Clermont-Ferrand",
    "LeMans",
    "Lorient",
    "Nantes_Saint-Nazaire",
    "Quimper",
    "Saint-Brieuc",
]

import dataloader

dataset, cm, earlystopping, weights, criterion = {}, {}, {}, {}, {}
datacode, status = [], []
for data in availabledata:
    print("load", data)
    dataset[data] = dataloader.SegSemDataset(root + data + "/train/")
    earlystopping[data] = dataset[data].getrandomtiles(128, 128, 16)
    cm[data] = np.zeros((2, 2), dtype=int)
    weights[data] = torch.Tensor([1, dataset[data].balance]).to(device)
    criterion[data] = torch.nn.CrossEntropyLoss(weight=weights[data])
    status.append(data in weaklysupervised)
    datacode.append(data)


print("train")


def accu(cm):
    return 100.0 * (cm[name][0][0] + cm[name][1][1]) / np.sum(cm[name])


def trainaccuracy(data):
    cm[data] = np.zeros((2, 2), dtype=int)
    net.eval()
    with torch.no_grad():
        for inputs, targets in earlystopping[data]:
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, pred = outputs.max(1)
            for i in range(pred.shape[0]):
                cm[data] += confusion_matrix(
                    pred[i].cpu().numpy().flatten(),
                    targets[i].cpu().numpy().flatten(),
                    labels=[0, 1],
                )


optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
nbepoch = 120
batchsize = 16

for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    print("preparing patch from each town")
    X = []
    Y = []
    for data in availabledata:
        code = [i for i in range(len(datacode)) if datacode[i] == data]
        code = code[0]

        XY = dataset[data].getrawrandomtiles(512, 128)
        for x, y in XY:
            X.append(torch.Tensor(np.transpose(x, axes=(2, 0, 1))).cpu())

            y = torch.from_numpy(y).long().cpu()
            tmp = torch.ones(y.shape)
            tmp * code

            ### ENCODE DATA SOURCE IN Y
            y = torch.stack([y, tmp])
            Y.append(y)

    X = torch.stack(X)  ### Kx3xWxH
    Y = torch.stack(Y)  ### Kx2xWxH

    patchdataset = torch.utils.data.TensorDataset(X, Y)
    patchloader = torch.utils.data.DataLoader(
        dataset, batch_size=batchsize, shuffle=True, num_workers=2
    )

    print("forward backward patch")
    net.train()
    for inputs, targets in patchloader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        _, pred = outputs.max(1)

        ### DECODE Y TO GET DATA SOURCE
        codes = [targets[i][1][0][0].cpu().numpy() for i in range(targets.shape[0])]
        targets = targets[:, 0, :, :]

        losses = torch.zeros(inputs.shape[0]).to(device)
        for i in range(preds.shape[0]):
            ### if supervised we want outputs==target
            ### if weakly supervised, we only want outputs*target==target
            if status[codes[i]]:
                targets[i] = targets[i] * preds[i]

            losses[i] = criterion[datacode[codes[i]]](outputs[i], targets[i])

        loss = torch.sum(losses)
        meanloss.append(loss.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    print("backup model")
    torch.save(net, "build/model.pth")

    for data in availabledata:
        trainaccuracy(data)
        print("accuracy", data, acc(cm[data]))

    allsupervised = [
        acc(cm[data]) for data in availabledata if data not in weaklysupervised
    ]
    if all([i > 98 for i in allsupervised]):
        print("training stops after reaching high training accuracy in each town")
        quit()
    globalcm = np.zeros((2, 2))
    for data in availabledata:
        if data not in weaklysupervised:
            globalcm += cm[data]
    print("global accuracy", acc(globalcm))
    if acc(globalcm) > 98:
        print("training stops after reaching high training accuracy in average")
        quit()

print("training stops after reaching time limit")
