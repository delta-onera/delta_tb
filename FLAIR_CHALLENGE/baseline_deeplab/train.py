import os
import torch
import torchvision
import dataloader

assert torch.cuda.is_available()

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/train/", "even")

print("define model")
net = dataloader.Mobilenet()
net = net.cuda()
net.eval()


print("train")


def crossentropy(y, z):
    tmp = torch.nn.CrossEntropyLoss(reduction="none")
    rawloss = tmp(z, y.long())
    return (rawloss).mean()


def dicelossi(y, z, i):
    eps = 0.00001
    z = z.softmax(dim=1)

    indexmap = torch.ones(z.shape).cuda()
    indexmap[:, i, :, :] = 0

    z0, z1 = z[:, i, :, :], (z * indexmap).sum(1)
    y0, y1 = (y == i).float(), (y != i).float()

    inter0, inter1 = (y0 * z0).sum(), (y1 * z1).sum()
    union0, union1 = (y0 + z1 * y0).sum(), (y1 + z0 * y1).sum()
    iou0, iou1 = (inter0 + eps) / (union0 + eps), (inter1 + eps) / (union1 + eps)
    iou = 0.5 * (iou0 + iou1)

    return 1 - iou


def diceloss(y, z):
    alldice = torch.zeros(13).cuda()
    for i in range(13):
        alldice[i] = dicelossi(y, z, i)
    return alldice.mean()


optimizer = torch.optim.Adam(net.parameters(), lr=0.0000001)
printloss = torch.zeros(1).cuda()
stats = torch.zeros((13, 13)).cuda()
batchsize = 8
nbbatchs = 2000
dataset.start()

for i in range(nbbatchs):
    x, y = dataset.getBatch(batchsize)
    x, y = x.cuda(), y.cuda()

    z = net(x)

    ce = crossentropy(y, z)
    dice = diceloss(y, z)
    loss = ce + dice

    with torch.no_grad():
        printloss += loss.clone().detach()
        _, z = z.max(1)
        stats += dataloader.confusion(y, z)

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
            stats = torch.zeros((13, 13)).cuda()
            print(i, "perf", perf)
            if perf[0] > 95:
                os._exit(0)

    if i > nbbatchs * 0.1:
        loss = loss * 0.7
    if i > nbbatchs * 0.2:
        loss = loss * 0.7
    if i > nbbatchs * 0.5:
        loss = loss * 0.7
    if i > nbbatchs * 0.8:
        loss = loss * 0.7

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
    optimizer.step()

os._exit(0)
