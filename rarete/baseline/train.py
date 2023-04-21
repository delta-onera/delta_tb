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
net = dataloader.RINET()
net = net.cuda()
net.train()


print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
printloss = torch.zeros(4).cuda()
nbbatchs = 10000
dataset.start()

for i in range(nbbatchs):
    x1, x2, m12 = dataset.getBatch()
    z1, z2 = net(x1.cuda()), net(x2.cuda())

    b1 = torch.nn.functional.relu(z1.abs() - 1)
    b2 = torch.nn.functional.relu(z2.abs() - 1)
    boundloss = (b1 * b1 + b2 * b2).mean()

    totalmean1, totalminmean1, _ = net.distance(z1)
    totalmean2, totalminmean2, _ = net.distance(z2)
    diffarealoss = -totalmean1 - totalmean2 - totalminmean1 - totalminmean2

    samearealoss, total = 0, 0
    for n in range(x1.shape[0]):
        for row in range(16):
            for col in range(16):
                q = numpy.asarray([row * 16 + 8, col * 16 + 8, 1])
                q = numpy.dot(m12[n], q)
                q = (int(q[0] / 16), int(q[1] / 16))
                if (0 <= q[0] < 16) and (0 <= q[1] < 16):
                    diff = z1[n, :, row, col] - z2[n, :, q[0], q[1]]
                    samearealoss = samearealoss + (diff ** 2).mean()
                    total += 1

    loss = boundloss + diffarealoss
    if total > 0:
        samearealoss = samearealoss / total
        loss = loss + samearealoss * 10

    with torch.no_grad():
        printloss[0] += loss.clone().detach()
        printloss[1] += boundloss.clone().detach()
        if total > 0:
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


os._exit(0)
