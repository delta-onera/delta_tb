import os
import torch
import dataloader

assert torch.cuda.is_available()

print("define model")
net = torch.load("build/fused.pth")
net = net.cuda()
net.eval()

print("load data")
dataset = dataloader.FLAIR2("train")


def crossentropy(y, z):
    class_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    tmp = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).cuda())
    return tmp(z, y.long())


def dicelossi(y, z, i):
    eps = 0.001
    z = z.softmax(dim=1)

    indexmap = torch.ones(z.shape).cuda()
    indexmap[:, i, :, :] = 0

    z0, z1 = z[:, i, :, :], (z * indexmap).sum(1)
    y0, y1 = (y == i).float(), (y != i).float()

    inter0, inter1 = (y0 * z0).sum(), (y1 * z1).sum()
    union0, union1 = (y0 + z1 * y0).sum(), (y1 + z0 * y1).sum()

    if union0 < eps or union1 < eps or union0 < inter0 or union1 < inter1:
        return 0

    iou0, iou1 = (inter0 + eps) / (union0 + eps), (inter1 + eps) / (union1 + eps)
    iou = 0.5 * (iou0 + iou1)
    return 1 - iou


def diceloss(y, z):
    alldice = torch.zeros(12).cuda()
    for i in range(12):
        alldice[i] = dicelossi(y, z, i)
    return alldice.mean()


print("train")

optimizer = torch.optim.Adam(net.parameters(), lr=0.000001)
printloss = [0, 0, 0, 0]
stats = torch.zeros((13, 13)).cuda()
nbbatchs = 20000
dataset.start()

batchsize = [32, 32, 32, 48, 16]
mode = 4

for i in range(nbbatchs):
    x, s, y = dataset.getBatch(batchsize[mode])
    x, s, y = x.cuda(), s.cuda(), y.cuda()

    TR = [j for j in range(y.shape[0]) if y[j][0][0] >= 0]
    if TR == []:
        i = i - 1
        continue

    if mode == 2:
        z = net(x, s, mode=2)
    else:
        z, semisup = net(x, s, mode=mode)

    dice = diceloss(y[TR], z[TR])
    ce = crossentropy(y[TR], z[TR])

    if mode == 2:
        loss = ce + dice
    else:
        loss = ce + dice + semisup

    with torch.no_grad():
        printloss[0] += float(loss)
        printloss[1] += float(ce)
        printloss[2] += float(dice)
        if mode != 2:
            printloss[3] += float(semisup)

        if TR != []:
            _, z = z.max(1)
            stats += dataloader.confusion(y[TR], z[TR])

        if i < 10:
            print(i, "/", nbbatchs, printloss)
        if i < 1000 and i % 100 == 99:
            print(i, "/", nbbatchs, [a / 100 for a in printloss])
            printloss = [0, 0, 0, 0]
        if i >= 1000 and i % 300 == 299:
            print(i, "/", nbbatchs, [a / 300 for a in printloss])
            printloss = [0, 0, 0, 0]

        if i % 1000 == 999:
            torch.save(net, "build/model.pth")
            perf = dataloader.perf(stats)
            stats = torch.zeros((13, 13)).cuda()
            print(i, "perf", perf)
            if perf[0] > 90:
                os._exit(0)

    if i % 100000 > nbbatchs * 0.1:
        loss = loss * 0.7
    if i % 100000 > nbbatchs * 0.2:
        loss = loss * 0.7
    if i % 100000 > nbbatchs * 0.5:
        loss = loss * 0.7
    if i % 100000 > nbbatchs * 0.8:
        loss = loss * 0.7

    optimizer.zero_grad()
    loss.backward()
    if mode != 4:
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    else:
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
    optimizer.step()

        
os._exit(0)
