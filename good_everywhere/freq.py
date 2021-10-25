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
import cropextractor
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
miniworld.start()

print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
if whereIam in ["ldtis706z", "wdtim719z"]:
    batchsize = 16
else:
    batchsize = 32
nbbatchs = 100000
printloss = torch.zeros(1).cuda()
stats = torch.zeros((len(miniworld.cities), 2, 2)).cuda()
for i in range(nbbatchs):
    if i % 500 > 70:
        perf = dataloader.perf(stats)

        perf = [(perf[j][0], j) for j in range(perf.shape[0])]
        perf = sorted(perf)

        priority = numpy.ones(len(perf))
        for j in range(stats.shape[0] // 2):
            priority[perf[j][1]] = 100

        x, y, batchchoise = miniworld.getbatch(batchsize, priority=priority)
    else:
        x, y, batchchoise = miniworld.getbatch(batchsize)
    x, y = x.cuda(), y.cuda()
    z = net(x)

    D = dataloader.distancetransform(y)
    nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
    weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
    CE = criterion(z, y.long())
    CE = torch.mean(CE * D)

    criteriondice = smp.losses.dice.DiceLoss(mode="multiclass")
    dice = criteriondice(z, y.long())
    loss = CE + 0.1 * dice

    with torch.no_grad():
        printloss += loss.clone().detach()
    if i > nbbatchs // 10:
        loss = loss * 0.5
    if i > nbbatchs // 5:
        loss = loss * 0.5
    if i > nbbatchs // 2:
        loss = loss * 0.5
    if i > (nbbatchs * 8) // 10:
        loss = loss * 0.5

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
    optimizer.step()

    with torch.no_grad():
        z = (z[:, 1, :, :] > z[:, 0, :, :]).float()
        for j in range(batchsize):
            cm = torch.zeros(2, 2).cuda()
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                cm[a][b] = torch.sum((z[j] == a).float() * (y[j] == b).float() * D[j])
            stats[batchchoise[j]] += cm

    if i % 100 == 99:
        print(i, "/50000", printloss / 100)
        printloss = torch.zeros(1).cuda()

    if i % 1000 == 999:
        torch.save(net, "build/model.pth")
        cm = torch.sum(stats, dim=0)
        perf = dataloader.perf(cm)
        print("perf", perf)
        if perf[0] > 92:
            print("training stops after reaching high training accuracy")
            os._exit(0)
        else:
            stats = torch.zeros((len(miniworld.cities), 2, 2)).cuda()

print("training stops after reaching time limit")
os._exit(0)
