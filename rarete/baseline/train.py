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
net = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
net[4] = torch.nn.Conv2d(96, 512, kernel_size=1)
net[5] = torch.nn.Identity()
net[6] = torch.nn.Identity()
net[7] = torch.nn.Identity()
net[8] = torch.nn.Identity()
net = net.cuda()
net.train()


print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
printloss = torch.zeros(5).cuda()
nbbatchs = 50000
dataset.start()

net.seuil = collections.deque(maxlen=1000)

for i in range(nbbatchs):
    x1, x2, m12 = dataset.getBatch()
    x1, x2 = (x1.cuda() - 0.5) * 2, (x2.cuda() - 0.5) * 2
    z1, z2 = net(x1), net(x2)

    b1 = torch.nn.functional.relu(z1.abs() - 1)
    b2 = torch.nn.functional.relu(z2.abs() - 1)
    boundloss = (b1 + b2 + b1 * b1 + b2 * b2).mean()

    N = z1.shape[0]
    diffarealoss, samearealoss, seuilloss = 0, 0, 0
    if len(net.seuil) > 500:
        targetseuil = sum(net.seuil) / len(net.seuil)
    else:
        targetseuil = None
    for n in range(N):
        distToOther, rows, cols, seuil = dataloader.distanceToAllOther(z1[n])
        diffarealoss = diffarealoss + distToOther

        net.seuil.append(seuil)
        if targetseuil is not None:
            seuilloss = seuilloss + (seuil - targetseuil) ** 2

        for j in range(rows.shape[0]):
            q = numpy.asarray([int(rows[i]), int(cols[i]), 0])
            q = 8 * q + 4
            q[-1] = 1
            q = numpy.dot(m12, q)
            q = (int(q[0]) // 8, int(q[1]) // 8)
            if 0 <= q[0] < 256 and 0 <= q[1] < 256:
                diff = z1[rows[i]][cols[i]] - z2[q[0]][q[1]]
                samearealoss = samearealoss + (diff ** 2).sum()

    loss = 5 * centredloss + 5 * samearealoss - diffarealoss + seuilloss

    with torch.no_grad():
        printloss[0] += loss.clone().detach()
        printloss[1] += centredloss.clone().detach()
        printloss[2] += samearealoss.clone().detach()
        printloss[3] += diffarealoss.clone().detach()
        printloss[4] += seuilloss.clone().detach()

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
