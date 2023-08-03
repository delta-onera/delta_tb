import os
import torch
import dataloader

assert torch.cuda.is_available()

print("define model")
originalnet = torch.load("../semisup2/build/model.pth")
originalnet = originalnet.cuda()
originalnet.eval()

net = torch.load("../semisup2/build/model.pth")
net = originalnet.cuda()
net.eval()
net.bfloat16()

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

optimizer = torch.optim.Adam(net.parameters(), lr=0.000005)
printloss = [0, 0, 0, 0]
stats = torch.zeros((13, 13)).cuda()
nbbatchs = 10000
dataset.start()

for i in range(nbbatchs):
    x, s, _ = dataset.getBatch(6)
    x, s = x.cuda(), s.cuda()

    with torch.no_grad():
        originalz, _ = originalnet(x, s)
        _, y = originalz.max(1)

    optimizer.zero_grad()

    with torch.autocast(device_type="gpu", dtype=torch.bfloat16):
        z, _ = net(x, s)
        dice = diceloss(y, z)
        ce = crossentropy(y, z)
        reg = ((z - originalz) ** 2).mean()
        loss = 0.5 * dice + 0.5 * ce + reg

    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
    optimizer.step()

    with torch.no_grad():
        printloss[0] += float(loss)
        printloss[1] += float(ce)
        printloss[2] += float(dice)
        printloss[2] += float(reg)

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

os._exit(0)
