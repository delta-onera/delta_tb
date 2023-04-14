import os
import numpy
import PIL
from PIL import Image
import torch
import torchvision
import dataloader
import collections

print("load data")
dataset = dataloader.getstdtraindataloader()

print("define model")
net = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
net[5] = torch.nn.Conv2d(128, 256, kernel_size=1)
del net[7], net[6]
net = net.cuda()
net.train()


print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
CE = torch.nn.CrossEntropyLoss()
printloss = torch.zeros(5).cuda()
nbbatchs = 50000
dataset.start()


def distanceToAllOther(X):
    D = X[:, :, None] - X[:, None, :]
    D = (D * D).sum(0)

    distTOother = D.mean()

    for i in range(D.shape[0]):
        D[i][i] += 100000
    _, v = D.min(1)
    seuil = sorted(list(v))[-5]

    amers = (v >= seuil).long()
    return distTOother, amers


for i in range(nbbatchs):
    x1, x2, m12 = dataset.getBatch()
    x1, x2 = (x1.cuda() - 0.5) * 2, (x2.cuda() - 0.5) * 2
    z1, z2 = net(x1), net(x2)
    p1, p2 = z1[:, 0:2, :, :], z2[:, 0:2, :, :]
    z1, z2 = z1[:, 2:, :, :], z2[:, 2:, :, :]

    b1 = torch.nn.functional.relu(z1.abs() - 1)
    b2 = torch.nn.functional.relu(z2.abs() - 1)
    boundloss = (b1 + b2 + b1 * b1 + b2 * b2).mean()

    N = z1.shape[0]
    diffarealoss, samearealoss, amerloss = 0, 0, 0
    for n in range(N):
        dist1, amer1 = distanceToAllOther(z1[n].reshape(254, -1))
        dist2, amer2 = distanceToAllOther(z2[n].reshape(254, -1))

        amerloss = amerloss + CE(p1[n].reshape(2, -1).transpose(0, 1), amer1)
        amerloss = amerloss + CE(p2[n].reshape(2, -1).transpose(0, 1), amer2)

        diffarealoss = diffarealoss + dist1 + dist2

        amer1 = amer1.reshape(16, 16)
        for row in range(amer1.shape[0]):
            for col in range(amer1.shape[1]):
                if amer1[row][col] == 0:
                    continue

                q = numpy.asarray([row * 8 + 4, col * 8 + 4, 1])
                q = numpy.dot(m12[n], q)
                q = (int(q[0] / 8), int(q[1] / 8))
                if (0 <= q[0] < 16) and (0 <= q[1] < 16):
                    diff = z1[n, :, row, col] - z2[n, :, q[0], q[1]]
                    samearealoss = samearealoss + (diff ** 2).sum()

    loss = 5 * boundloss + 10 * samearealoss - diffarealoss + amerloss

    with torch.no_grad():
        printloss[1] += boundloss.clone().detach()
        if samearealoss != 0:
            printloss[2] += samearealoss.clone().detach()
        printloss[3] += diffarealoss.clone().detach()
        printloss[4] += amerloss.clone().detach()
        printloss[0] += loss.clone().detach()

        if i % 100 == 99:
            print(i, printloss.cpu() / 100)
            printloss = torch.zeros(5).cuda()
        if i % 1000 == 999:
            torch.save(net, "build/model.pth")

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
