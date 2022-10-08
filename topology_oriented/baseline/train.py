import os
import torch
import torchvision

assert torch.cuda.is_available()

sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
sys.path.append("/d/achanhon/github/pytorch-image-models")
sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import miniworld


print("load data")
dataset = miniworld.Simple("/scratchf/miniworld/potsdam/train/")

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
printloss = torch.zeros(1).cuda()
stats = torch.zeros((2, 2)).cuda()
batchsize = 32
nbbatchs = 75000
dataset.start()


def crossentropy(y, z, D):
    tmp = torch.nn.CrossEntropyLoss(reduction="none")
    rawloss = tmp(z, y.long())
    return (rawloss * D).mean()


def diceloss(y, z, D):
    eps = 0.00001
    z = z.softmax(dim=1)
    z0, z1 = z[:, 0, :, :], z[:, 1, :, :]
    y0, y1 = (y == 0).float(), (y == 1).float()

    inter0, inter1 = (y0 * z0 * D).sum(), (y1 * z1 * D).sum()
    union0, union1 = ((y0 + z1 * y0) * D).sum(), ((y1 + z0 * y1) * D).sum()
    iou0, iou1 = (inter0 + eps) / (union0 + eps), (inter1 + eps) / (union1 + eps)
    iou = 0.5 * (iou0 + iou1)

    return 1 - iou


for i in range(nbbatchs):
    x, y = dataset.getBatch(batchsize)
    x, y, D = x.cuda(), y.cuda(), torch.ones(y.shape).cuda()

    z = net(x)

    CE = crossentropy(y, z, D)
    dice = diceloss(y, z, D)
    loss = CE + dice

    with torch.no_grad():
        printloss += loss.clone().detach()
        z = (z[:, 1, :, :] > z[:, 0, :, :]).clone().detach().float()
        stats += miniworld.confusion(y[j], z[j], D)

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
            perf = miniworld.perf(stats)
            stats = torch.zeros((2, 2)).cuda()
            print(i, "perf", perf)
            if perf > 0.95:
                os._exit(0)

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

os._exit(0)
