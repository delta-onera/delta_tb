import os
import numpy
import PIL
from PIL import Image
import torch
import torchvision
import dataloader

print("load data")
dataset = dataloader.getstdtraindataloader()

print("define model")
net = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
net = net.cuda()
net.train()


print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
printloss = torch.zeros(4).cuda()
nbbatchs = 50000
dataset.start()

for i in range(nbbatchs):
    x = dataset.getBatch()
    x = ((x / 255) - 0.5) * 2
    x1, x2 = x[:, 0:3, :, :].cuda(), x[:, 3:6, :, :].cuda()

    z1 = net(x1)
    z2 = net(x2)

    centredloss = torch.nn.functional(z1.abs() + z2.abs() - 1)
    centredloss = (centredloss + centredloss ** 2).mean()

    samearealoss = ((z1 - z2) ** 2).mean()

    N = x.shape[0]
    diffarealoss1 = [(z1[i] - z1[i + 1]).abs() for i in range(N - 1)]
    diffarealoss2 = [(z2[i] - z2[i + 4]).abs() for i in range(N - 4)]
    diffarealoss = diffarealoss1 + diffarealoss2
    diffarealoss = sum(diffarealoss) / len(diffarealoss)

    loss = centredloss + samearealoss + diffarealoss

    with torch.no_grad():
        printloss[0] += loss.clone().detach()
        printloss[1] += centredloss.clone().detach()
        printloss[2] += samearealoss.clone().detach()
        printloss[3] += diffarealoss.clone().detach()

        if i % 100 == 99:
            print(i, printloss.cpu() / 100)
            printloss = torch.zeros(4).cuda()
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
