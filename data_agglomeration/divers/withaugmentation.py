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

outputname = "model.pth"
if len(sys.argv) > 1:
    outputname = sys.argv[1]
os.system("cat withaugmentation.py")

whereIam = os.uname()[1]

print("define model")
if whereIam == "wdtim719z":
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import collections
import random

tmp = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=128,
)
net = torch.nn.Sequential(tmp, torch.nn.LeakyReLU(), torch.nn.Conv2d(128, 2, 1))
net = net.cuda()
net.train()


print("load data")
import dataloader

miniworld = dataloader.MiniWorld()

earlystopping = miniworld.getrandomtiles(5000, 128, 32)
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
nbepoch = 800
batchsize = 32

import dependencyfreeimgaug

for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = miniworld.getrandomtiles(10000, 128, batchsize)
    for x, y in XY:
        x, y = x.to(device), y.to(device)

        # x = dependencyfreeimgaug.augment(x)

        preds = net(x)
        loss = criterion(preds, y)
        meanloss.append(loss.cpu().data.numpy())

        if epoch > 100:
            loss = loss * 0.5
        if epoch > 200:
            loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    print("backup model")
    torch.save(net, "build/" + outputname)
    cm = trainaccuracy()
    print("accuracy", accu(cm))

    if accu(cm) > 98:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")
