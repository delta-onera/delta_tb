import os
import torch
import torchvision
import digitanieV2

assert torch.cuda.is_available()

print("load data")
dataset = digitanieV2.getDIGITANIE("even")

print("define model")
net = miniworld.Mobilenet()
net = net.cuda()
net.train()


print("train")


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


optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
printloss = torch.zeros(1).cuda()
stats = torch.zeros((2, 2)).cuda()
batchsize = 8
nbbatchs = 100000
dataset.start()

for i in range(nbbatchs):
    x, y = dataset.getBatch(batchsize)
    x, y = x.cuda(), y.cuda()
    skeleton = digitanieV2.compute(y)
    D = y + 100 * (y == 0).float() + 500 * skeleton

    z = net(x)

    ce = crossentropy(y, z, D)
    dice = diceloss(y, z, D)
    loss = ce + dice * 0.1

    with torch.no_grad():
        printloss += loss.clone().detach()
        z = (z[:, 1, :, :] > z[:, 0, :, :]).clone().detach().float()
        stats += digitanieV2.confusion(y, z)

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
            perf = digitanieV2.perf(stats)
            stats = torch.zeros((2, 2)).cuda()
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
