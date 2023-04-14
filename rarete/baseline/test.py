import os
import numpy
import PIL
from PIL import Image
import torch
import dataloader


def torchTOpil(x):
    visu = numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))
    return PIL.Image.fromarray(numpy.uint8((visu + 1) * 125))


def drawrect(x, c):
    c = (c[0] * 8 + 4, c[1] * 8 + 4)
    x[0, c[0] - 3, c[1] - 3 : c[1] + 3] = 1
    x[0, c[0] - 3, c[1] + 3 : c[1] + 3] = 1
    x[0, c[0] - 3 : c[0] + 3, c[1] - 3] = 1
    x[0, c[0] - 3 : c[0] + 3, c[1] + 3] = 1


print("load data")
dataset = dataloader.getstdtestdataloader()

print("define model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()


print("test")
dataset.start()

totalmatch, goodmatch, perfectmatch = 0, 0, 0
with torch.no_grad():
    for k in range(10):
        x1, x2, m12 = dataset.getBatch()
        x1, x2 = (x1.cuda() - 0.5) * 2, (x2.cuda() - 0.5) * 2
        z1, z2 = net(x1), net(x2)
        p1, p2 = z1[:, 0:2, :, :], z2[:, 0:2, :, :]
        z1, z2 = z1[:, 2:, :, :], z2[:, 2:, :, :]

        amers1, amers2 = [], []
        for row in range(16):
            for col in range(16):
                if p1[1, row, col] > p1[0, row, col]:
                    amers1.append((row, col))
                if p2[1, row, col] > p2[0, row, col]:
                    amers2.append((row, col))

        # only debug for the moment
        for c in amers1:
            drawrect(x1, c)
        for c in amers2:
            drawrect(x2, c)

        visu1, visu2 = torchTOpil(x1), torchTOpil(x2[I[0][1]])
        visu1.save("build/" + str(k) + "_1.png")
        visu2.save("build/" + str(k) + "_2.png")

        visu1, visu2 = torchTOpil(x1[I[20][1]]), torchTOpil(x2[I[20][1]])
        visu1.save("build/" + str(k) + "_3.png")
        visu2.save("build/" + str(k) + "_4.png")


os._exit(0)
