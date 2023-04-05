import os
import numpy
import PIL
from PIL import Image
import torch
import dataloader

torch.backends.cudnn.benchmark = True


def torchTOpil(x):
    visu = numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))
    return PIL.Image.fromarray(visu)


print("load data")
dataset = dataloader.getstdtestdataloader()

print("define model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()


print("test")
dataset.start()

with torch.no_grad():
    for k in range(10):
        x = dataset.getBatch()
        x = ((x / 255) - 0.5) * 2
        x1, x2 = x[:, 0:3, :, :].cuda(), x[:, 3:6, :, :].cuda()

        z1 = net(x1)
        z2 = net(x2)

        N = x.shape[0]
        D = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                D[i][j] = (z1[i] - z2[j]) ** 2
        D_ = D.clone()
        for i in range(N):
            D_[i][i] += 100000

        I = [(D[i][i] - D_[i].min(), i) for i in range(N)]
        I = sorted(I)

        x1, x2 = torchTOpil(x1[I[0][1]]), torchTOpil(x2[I[0][1]])
        x1.save("build/" + str(k) + "_1.png")
        x2.save("build/" + str(k) + "_2.png")

        x1, x2 = torchTOpil(x1[I[20][1]]), torchTOpil(x2[I[20][1]])
        x1.save("build/" + str(k) + "_3.png")
        x2.save("build/" + str(k) + "_4.png")


os._exit(0)
