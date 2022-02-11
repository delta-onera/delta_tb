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
dataset = dataloader.CropExtractor(sys.argv[1])

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
weights = torch.Tensor([1, 10]).cuda()
criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
printloss = torch.zeros(1).cuda()
stats = torch.zeros((len(dataset.cities), 2, 2)).cuda()
batchsize = 32
nbbatchs = 400000
dataset.start()


def diceloss(y, z, D):
    eps = 1e-7
    y = torch.nn.functional.one_hot(y, num_classes=2)
    y = y.transpose(2, 3).transpose(1, 2).float()
    z = z.log_softmax(dim=1).exp()

    intersection = torch.sum(y * z * D)
    cardinality = torch.sum((y + z) * D).clamp_min(eps)
    return (2.0 * intersection + eps) / (cardinality + eps)


for i in range(nbbatchs):
    x, y, batchchoise, goodlabel = dataset.getBatch(batchsize)
    x, y = x.cuda(), y.cuda()
    z = net(x)

    D = dataloader.distancetransform(y.float())
    batchchoise, goodlabel = batchchoise.cuda(), goodlabel.cuda()

    tmp = D * goodlabel.unsqueeze(1).unsqueeze(1)
    CE = criterion(z, y)
    CE = torch.mean(CE * tmp)

    tmp = torch.stack([tmp, tmp], dim=1)
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
            perf = dataloader.perf(stats)
            print(i, "perf", perf)
            if perf[0] > 92:
                print("training stops after reaching high training accuracy")
                os._exit(0)
            else:
                stats = torch.zeros((len(dataset.cities), 2, 2)).cuda()

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
