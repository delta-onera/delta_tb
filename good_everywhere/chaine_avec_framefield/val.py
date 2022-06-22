import os
import sys
import numpy
import PIL
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import util

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()

sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
sys.path.append("/d/achanhon/github/pytorch-image-models")
sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import miniworld

print("load data")
miniworlddataset = miniworld.MiniWorld("/test/")

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()


print("test")


def largeforward(net, image, tilesize=128, stride=64):
    pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).cuda()
    image = image.cuda()
    for row in range(0, image.shape[2] - tilesize + 1, stride):
        for col in range(0, image.shape[3] - tilesize + 1, stride):
            tmp = net(image[:, :, row : row + tilesize, col : col + tilesize])
            pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


globalcm, globalcm1 = torch.zeros((2, 2)).cuda(), torch.zeros((2, 2)).cuda()
with torch.no_grad():
    for k, city in enumerate(miniworlddataset.cities):
        print(k, city)
        cm, cm1 = torch.zeros((2, 2)).cuda(), torch.zeros((2, 2)).cuda()

        for i in range(miniworlddataset.data[city].NB):
            x, y = miniworlddataset.data[city].getImageAndLabel(i, torchformat=True)
            x, y = x.cuda(), y.cuda().float()

            h, w = y.shape[0], y.shape[1]
            globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
            power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
            x = power2resize(x)

            z = largeforward(net, x.unsqueeze(0))
            z = globalresize(z)
            z = (z[0, 1, :, :] > z[0, 0, :, :]).float()

            border = util.getborder(y)
            border = 1 - border

            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                cm[a][b] += torch.sum((z == a).float() * (y == b).float())
                cm1[a][b] += torch.sum((z == a).float() * (y == b).float() * border)
            globalcm += cm
            globalcm1 += cm1

            if len(os.listdir("build")) < 100:
                nextI = len(os.listdir("build"))
                debug = util.torchTOpil(globalresize(x))
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_x.png")
                debug = (2.0 * y - 1) * border * 127 + 127
                debug = debug.cpu().numpy()
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_y.png")
                debug = z.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_z.png")

        print("perf0=", util.perf(cm))
        print("perf1=", util.perf(cm1))
        print("bords=", util.perf(cm - cm1))


print("global result")
print("perf0=", util.perf(globalcm))
print("perf1=", util.perf(globalcm1))
print("bords=", util.perf(globalcm - globalcm1))
