import os
import numpy
import torch
import torchvision
import dataloader


def drawrect(x, c):
    c = (c[0] * 16 + 8, c[1] * 16 + 8)
    x[0, c[0] - 3, c[1] - 3 : c[1] + 3] = 1
    x[0, c[0] - 3, c[1] + 3 : c[1] + 3] = 1
    x[0, c[0] - 3 : c[0] + 3, c[1] - 3] = 1
    x[0, c[0] - 3 : c[0] + 3, c[1] + 3] = 1
    return x


print("load data")
dataset = dataloader.getstdtestdataloader()

print("define model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()


print("test")
dataset.start()

totalmatch, goodmatch, perfectmatch = 0, 0, 0
if True:
    x1, x2, m12 = dataset.getBatch()
    z1, z2 = net(x1.cuda()), net(x2.cuda())

    _, _, farestX1 = net.distance(z1)
    _, _, farestX2 = net.distance(z2)

    for n in range(x1.shape[0]):
        for row in range(16):
            for col in range(16):
                if farestX1[n][row][col]:
                    x1[n] = drawrect(x1[n], (row, col))
                if farestX2[n][row][col]:
                    x2[n] = drawrect(x2[n], (row, col))

        torchvision.utils.save_image(x1[n], "build/" + str(n) + "_1.png")
        torchvision.utils.save_image(x2[n], "build/" + str(n) + "_2.png")

os._exit(0)
