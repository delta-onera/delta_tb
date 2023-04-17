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
net.f = torch.nn.Conv2d(128, 128, kernel_size=1)
net.p = torch.nn.Conv2d(128, 2, kernel_size=1)
del net[7], net[6], net[5]
net = net.cuda()
net.train()


print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
printloss = torch.zeros(4).cuda()
nbbatchs = 1000
dataset.start()

CE = torch.nn.CrossEntropyLoss()

for i in range(nbbatchs):
    x1, x2, m12 = dataset.getBatch()
    x1, x2 = (x1.cuda() - 0.5) * 2, (x2.cuda() - 0.5) * 2
    z1, z2 = net.f(net(x1)), net.f(net(x2))

    b1 = torch.nn.functional.relu(z1.abs() - 1)
    b2 = torch.nn.functional.relu(z2.abs() - 1)
    boundloss = (b1 + b2 + b1 * b1 + b2 * b2).mean()

    N = z1.shape[0]
    diffarealoss, samearealoss, total = 0, 0, 0
    for n in range(N):
        Z = z1[n].reshape(128, -1)
        D1 = Z[:, :, None] - Z[:, None, :]
        D1 = D1.abs().mean() / N

        Z = z2[n].reshape(128, -1)
        D2 = Z[:, :, None] - Z[:, None, :]
        D2 = D2.abs().mean() / N
        diffarealoss = diffarealoss - D1 - D2

        for row in range(amer1.shape[0]):
            for col in range(amer1.shape[1]):
                q = numpy.asarray([row * 16 + 8, col * 16 + 8, 1])
                q = numpy.dot(m12[n], q)
                q = (int(q[0] / 16), int(q[1] / 16))
                if (0 <= q[0] < 16) and (0 <= q[1] < 16):
                    diff = z1[n, :, row, col] - z2[n, :, q[0], q[1]]
                    samearealoss = samearealoss + (diff ** 2).mean()
                    total += 1

    loss = boundloss - diffarealoss
    if total > 0:
        loss = loss + samearealoss / total * 10

    with torch.no_grad():
        printloss[0] += loss.clone().detach()
        printloss[1] += boundloss.clone().detach()
        if total != 0:
            printloss[2] += samearealoss.clone().detach()
        printloss[3] += diffarealoss.clone().detach()

        if i % 10 == 9:
            print(i, printloss.cpu() / 10)
            printloss = torch.zeros(4).cuda()
        if i % 100 == 99:
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

print("train second head")
optimizer = torch.optim.Adam(net.p.parameters(), lr=0.0001)
printloss = 0
CE = torch.nn.CrossEntropyLoss()
nbbatchs = 100
for i in range(nbbatchs):
    x1, _, _ = dataset.getBatch()
    with torch.no_grad():
        x1 = (x1.cuda() - 0.5) * 2
        c1 = net(x1)
        z1 = net.f(c1)
        amers = torch.zeros(N, 16, 16)

    for n in range(N):
        Z = z[n].reshape(128, -1)
        D = Z[:, :, None] - Z[:, None, :]
        D = (D ** 2).mean(0)
        for i in range(D.shape[0]):
            D[i][i] = 10000
            v, _ = D.min(1)
            seuil = sorted(list(v))[-5]

        amers[n] = (v >= seuil).reshape(16, 16)

    amers = amers.long()
    p1 = net.p(c1)
    loss = CE(p1, amers)

    with torch.no_grad():
        printloss += loss.clone().detach()

        if i % 10 == 9:
            print(i, printloss.cpu() / 10)
            printloss = 0
        if i % 100 == 99:
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
