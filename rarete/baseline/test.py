import os
import numpy
import torch
import torchvision
import dataloader


def drawrect(x, c, i=0):
    I = [0, 1, 2]
    if i == 0:
        I = 0
    if i == 1:
        I = 0
    if i == 2:
        I = 0
    if i == 3:
        I = [0, 1]
    if i == 4:
        I = [0, 2]
    if i == 6:
        I = [1, 2]

    c = (c[0] * 16 + 8, c[1] * 16 + 8)
    x[I, c[0] - 3, c[1] - 3 : c[1] + 3] = 1
    x[I, c[0] - 3, c[1] + 3 : c[1] + 3] = 1
    x[I, c[0] - 3 : c[0] + 3, c[1] - 3] = 1
    x[I, c[0] - 3 : c[0] + 3, c[1] + 3] = 1
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

    for n in range(x1.shape[0]):
        i = 0
        for row in range(16):
            for col in range(16):
                if farestX1[n][row][col]:
                    drawrect(x1[n], (row, col), i)

                    diff = z2[n] - z1[n, :, row, col].unsqueeze(-1).unsqueeze(-1)
                    diff = (diff ** 2).sum(0)
                    _, idx = torch.min(diff.reshape(16 * 16), 0)
                    row_index, col_index = int(idx) // 16, int(idx) % 16

                    drawrect(x2[n], (row_index, col_index), i)

                    q = numpy.asarray([row * 16 + 8, col * 16 + 8, 1])
                    q = numpy.dot(m12, q)
                    q = (int(q[0] / 16), int(q[1] / 16))

                    # diff q and row_index, col_index
                    totalmatch += 1
                    if q[0] == row_index and q[1] == col_index:
                        perfectmatch += 1
                        goodmatch += 1
                    if (q[0] - row_index) ** 2 + (q[1] - col_index) ** 2 <= 4:
                        goodmatch += 1

        torchvision.utils.save_image(x1[n], "build/" + str(n) + "_1.png")
        torchvision.utils.save_image(x2[n], "build/" + str(n) + "_2.png")

    # matching and quantitative eval
    print(totalmatch, goodmatch, perfectmatch)

os._exit(0)
