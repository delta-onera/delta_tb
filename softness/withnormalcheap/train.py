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

sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
sys.path.append("/d/achanhon/github/pytorch-image-models")
sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import noisyairs

print("load data")
dataset = noisyairs.AIRS("/train/")

print("define model")
net = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
net = net.cuda()
net.train()


print("train")
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
printloss = torch.zeros(2).cuda()
stats = torch.zeros((2, 2)).cuda()
batchsize = 32
nbbatchs = 75000
dataset.start()
sobel = noisyairs.Sobel()


def diceloss(y, z, D):
    eps = 0.00001
    z = z.log_softmax(dim=1).exp()
    z0, z1 = z[:, 0, :, :], z[:, 1, :, :]
    y0, y1 = (y == 0).float(), (y == 1).float()

    inter0, inter1 = (y0 * z0 * D).sum(), (y1 * z1 * D).sum()
    union0, union1 = ((y0 + z1 * y0) * D).sum(), ((y1 + z0 * y1) * D).sum()
    iou0, iou1 = (inter0 + eps) / (union0 + eps), (inter1 + eps) / (union1 + eps)
    iou = 0.5 * (iou0 + iou1)

    return 1 - iou


for i in range(nbbatchs):
    x, y = dataset.getBatch(batchsize)
    x, y = x.cuda(), y.cuda()
    z = net(x)

    yy, border = sobel(torch.unsqueeze(2.0 * y - 1.0, dim=1))

    CE = criterion(z, y)
    CE = torch.mean(CE * (1 - border))
    dice = diceloss(y, z, 1 - border)

    zz, _ = sobel(torch.unsqueeze(z[:, 1, :, :] - z[:, 0, :, :], dim=1))

    gradientdiff = torch.sum(zz * yy, dim=1)
    gradientdiff = torch.mean((1 - gradientdiff) * border)

    loss = (CE + dice) + 10 * gradientdiff

    with torch.no_grad():
        printloss[0] += (CE + dice).clone().detach()
        printloss[1] += gradientdiff.clone().detach()
        z = (z[:, 1, :, :] > z[:, 0, :, :]).clone().detach().float()
        for j in range(batchsize):
            stats += noisyairs.confusion(y[j], z[j], size=1)

        if i < 10:
            print(i, "/", nbbatchs, printloss)
        if i < 1000 and i % 100 == 99:
            print(i, "/", nbbatchs, printloss / 100)
            printloss = torch.zeros(2).cuda()
        if i >= 1000 and i % 300 == 299:
            print(i, "/", nbbatchs, printloss / 300)
            printloss = torch.zeros(2).cuda()

        if i % 1000 == 999:
            torch.save(net, "build/model.pth")
            perf = noisyairs.perf(stats)
            print(i, "perf", perf)
            stats = torch.zeros((2, 2)).cuda()

    if i > nbbatchs * 0.1:
        loss = loss * 0.5
    if i > nbbatchs * 0.2:
        loss = loss * 0.5
    if i > nbbatchs * 0.5:
        loss = loss * 0.5
    if i > nbbatchs * 0.8:
        loss = loss * 0.5

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
    optimizer.step()

print("training stops after reaching time limit")
os._exit(0)
