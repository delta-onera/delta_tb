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
net = torch.load("build/model.pth")
net = net.cuda()
net.train()


print("train second head")
optimizer = torch.optim.Adam(net.p_.parameters(), lr=0.0001)
printloss = 0
CE = torch.nn.CrossEntropyLoss()
nbbatchs = 1000
dataset.start()

for i in range(nbbatchs):
    x1, x2, _ = dataset.getBatch()
    N = x1.shape[0]
    with torch.no_grad():
        z1, z2 = net(x1.cuda()), net(x2.cuda())
        f1, f2 = net.f_(z1), net.f_(z2)
        amers1 = torch.zeros(N, 16, 16).cuda()
        amers2 = torch.zeros(N, 16, 16).cuda()

    for n in range(N):
        Z = z1[n].reshape(128, -1)
        D = Z[:, :, None] - Z[:, None, :]
        D = D.abs().mean(0)
        for j in range(16):
            D[j][j] = 10000
            v, _ = D.min(1)
            seuil = sorted(list(v))[-5]
        amers1[n] = (v >= seuil).reshape(16, 16)

        Z = z2[n].reshape(128, -1)
        D = Z[:, :, None] - Z[:, None, :]
        D = D.abs().mean(0)
        for j in range(16):
            D[j][j] = 10000
            v, _ = D.min(1)
            seuil = sorted(list(v))[-5]
        amers2[n] = (v >= seuil).reshape(16, 16)

    amers1 = torch.nn.functional.interpolate(amers1.unsqueeze(1), (256, 256))
    torchvision.utils.save_image(amers1, "build/wtf_x.png")
    torchvision.utils.save_image(x1, "build/wtf_y.png")
    quit()

    amers1, amers2 = amers1.long(), amers2.long()
    p1, p2 = net.p_(z1), net.p_(z2)
    loss = CE(p1, amers1) + CE(p2, amers2)

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