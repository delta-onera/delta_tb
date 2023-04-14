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
    c = (c[0] * 16 + 8, c[1] * 16 + 8)
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
    if True:
        x1, x2, m12 = dataset.getBatch()
        x1, x2 = (x1.cuda() - 0.5) * 2, (x2.cuda() - 0.5) * 2
        z1, z2 = net(x1), net(x2)
        p1, p2 = z1[:, 0:2, :, :], z2[:, 0:2, :, :]
        z1, z2 = z1[:, 2:, :, :], z2[:, 2:, :, :]

        for i in range(x1.shape[0]):
            amers1, amers2 = [], []
            for row in range(16):
                for col in range(16):
                    if p1[i, 1, row, col] > p1[i, 0, row, col]:
                        amers1.append((row, col))
                    if p2[i, 1, row, col] > p2[i, 0, row, col]:
                        amers2.append((row, col))

            # only debug for the moment
            for c in amers1:
                drawrect(x1[i], c)
            for c in amers2:
                drawrect(x2[i], c)

            visu1, visu2 = torchTOpil(x1[i]), torchTOpil(x2[i])
            visu1.save("build/" + str(i) + "_1.png")
            visu2.save("build/" + str(i) + "_2.png")


os._exit(0)
