import os
import numpy
import PIL
from PIL import Image
import torch
import dataloader


def torchTOpil(x):
    visu = numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))
    return PIL.Image.fromarray(numpy.uint8((visu + 1) * 125))


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

        N = x.shape[0]
        z1 = net(x1).view(N, 1280, -1).mean(2)
        z2 = net(x2).view(N, 1280, -1).mean(2)

        D = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                D[i][j] = ((z1[i] - z2[j]) ** 2).mean()
        D_ = D.clone()
        for i in range(N):
            D_[i][i] += 100000

        I = [(D[i][i] < D_[i].min()) for i in range(N)]
        print(len(I) / N)
        I = [(D[i][i] - D_[i].min(), i) for i in range(N)]
        I = sorted(I)

        visu1, visu2 = torchTOpil(x1[I[0][1]]), torchTOpil(x2[I[0][1]])
        visu1.save("build/" + str(k) + "_1.png")
        visu2.save("build/" + str(k) + "_2.png")

        visu1, visu2 = torchTOpil(x1[I[20][1]]), torchTOpil(x2[I[20][1]])
        visu1.save("build/" + str(k) + "_3.png")
        visu2.save("build/" + str(k) + "_4.png")


os._exit(0)
