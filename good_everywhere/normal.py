import os
import sys
import numpy
import PIL
from PIL import Image
import torch
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()

whereIam = os.uname()[1]
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
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import dataloader

print("define model")
net = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
net = net.cuda()
net.train()


print("load data")
miniworld = dataloader.MiniWorld(flag="train")
miniworld.openpytorchloader()

print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
nbepoch = 400
batchsize = 128
for i in range(500000):
    stats = torch.zeros(3).cuda()
    for x, y in miniworld.getbatch(batchsize):
        x, y = x.cuda(), y.cuda()
        z = net(x)

        D = dataloader.distancetransform(y)
        nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
        weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
        criteriondice = smp.losses.dice.DiceLoss(mode="multiclass")
        CE = criterion(z, y)
        CE = torch.mean(CE * D)
        dice = criteriondice(z, y)
        loss = CE + dice

        with torch.no_grad():
            stats[0] += loss.clone().detach()
        if i > 100000:
            loss = loss * 0.5
        if i > 200000:
            loss = loss * 0.5
        if i > 300000:
            loss = loss * 0.5
        if i > 400000:
            loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
        optimizer.step()

        with torch.no_grad():
            z = (z[:, 1, :, :] > z[:, 0, :, :]).long()
            stats[1] += torch.sum((z == y).float() * D)
            stats[2] += torch.sum(D)

        if i % 61 == 60:
            print(i, "/500000", stats[0] / 61)

        if i % 500 == 499:
            torch.save(net, "build/model.pth")
            print("accuracy", 100 * stats[1] / stats[2])
            if 100 * stats[1] / stats[2]:
                print("training stops after reaching high training accuracy")
                quit()
            else:
                stats = torch.zeros(3).cuda()

print("training stops after reaching time limit")
