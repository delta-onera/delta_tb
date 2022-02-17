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
import dataloader

print("load data")
miniworld = dataloader.MiniWorld("/train/")

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
weights = torch.Tensor([1, 1]).cuda()
criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
printloss = torch.zeros(1).cuda()
stats = torch.zeros((len(miniworld.cities), 2, 2)).cuda()
batchsize = 32
nbbatchs = 400000
miniworld.start()


def diceloss(y, z, D):
    eps = 1e-7
    y = torch.nn.functional.one_hot(y, num_classes=2)
    y = y.transpose(2, 3).transpose(1, 2).float()
    z = z.log_softmax(dim=1).exp()

    intersection = y * z
    cardinality = y + z
    iou = (2 * intersection + eps) / (cardinality + eps)
    return -torch.mean(iou * D)


for i in range(nbbatchs):
    x, y, batchchoise, _ = miniworld.getBatch(batchsize)
    x, y, batchchoise = x.cuda(), y.cuda(), batchchoise.cuda()
    z = net(x)

    D = dataloader.distancetransform(y.float())
    y0, y1 = (y == 0).float(), (y == 1).float()
    D = D * y1 + y0

    CE = criterion(z, y)
    CE = torch.mean(CE * D)

    tmp = torch.stack([D, D], dim=1)
    dice = diceloss(y, z, tmp)
    loss = CE + dice

    with torch.no_grad():
        printloss += loss.clone().detach()
        z = (z[:, 1, :, :] > z[:, 0, :, :]).clone().detach().float()
        for j in range(batchsize):
            cm = torch.zeros(2, 2).cuda()
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                cm[a][b] = torch.sum((z[j] == a).float() * (y[j] == b).float() * D[j])
            stats[batchchoise[j]] += cm

        if i < 10:
            print(i, "/", nbbatchs, printloss)
        if i < 1000 and i % 100 == 99:
            print(i, "/", nbbatchs, printloss / 100)
            printloss = torch.zeros(1).cuda()
        if i >= 1000 and i % 300 == 299:
            print(i, "/", nbbatchs, printloss / 300)
            printloss = torch.zeros(1).cuda()

        if i % 1000 == 999:
            torch.save(net, "build/model.pth")
            perf = dataloader.perf(torch.sum(stats, dim=0))
            print(i, "perf", perf)
            if perf[0] > 92:
                print("training stops after reaching high training accuracy")
                os._exit(0)
            else:
                stats = torch.zeros((len(miniworld.cities), 2, 2)).cuda()

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
