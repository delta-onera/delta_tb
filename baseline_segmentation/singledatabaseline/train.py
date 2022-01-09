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

sys.path.append("/home/achanhon/github/EfficientNet-PyTorch")
sys.path.append("/home/achanhon/github/pytorch-image-models")
sys.path.append("/home/achanhon/github/pretrained-models.pytorch")
sys.path.append("/home/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import cropextractor

print("load data")
dataset = cropextractor.CropExtractor(sys.argv[1] + "/train/")

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
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
batchsize = 16
nbbatchs = 50000
printloss = torch.zeros(1).cuda()
stats = torch.zeros((2, 2)).cuda()
dataset.start()

for i in range(nbbatchs):
    x, y = dataset.getBatch(batchsize)
    x, y = x.cuda(), y.cuda()
    z = net(x)

    D = cropextractor.distancetransform(y)
    nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
    weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
    CE = criterion(z, y.long())
    CE = CE * D
    CE = torch.mean(CE)

    criteriondice = smp.losses.dice.DiceLoss(mode="multiclass")
    dice = criteriondice(z, y.long())
    loss = CE + 0.1 * dice

    with torch.no_grad():
        printloss += loss.clone().detach()
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

    with torch.no_grad():
        z = (z[:, 1, :, :] > z[:, 0, :, :]).float()
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            stats[a][b] += torch.sum((z == a).float() * (y == b).float() * D)

    if i % 100 == 99:
        print(i, "/200000", printloss / 100)
        printloss = torch.zeros(1).cuda()

    if i % 1000 == 999:
        torch.save(net, "build/model.pth")
        perf = cropextractor.perf(stats)
        print("perf", perf)
        if perf[0] > 92:
            print("training stops after reaching high training accuracy")
            os._exit(0)
        else:
            stats = torch.zeros((2, 2)).cuda()

print("training stops after reaching time limit")
os._exit(0)
