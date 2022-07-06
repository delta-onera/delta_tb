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

whereIam = os.uname()[1]
if whereIam == "ldtis706z":
    sys.path.append("/home/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/pytorch-image-models")
    sys.path.append("/home/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/home/achanhon/github/segmentation_models.pytorch")
if whereIam == "wdtim719z":
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")
if whereIam in ["calculon", "astroboy", "flexo", "bender", "baymax"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import digitanie

print("load data")
dataset = digitanie.DigitanieALL()

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()


print("test")
globalcm, globalcm1 = torch.zeros((2, 2)).cuda(), torch.zeros((2, 2)).cuda()
with torch.no_grad():
    for city in dataset.cities:
        print(city)
        cm, cm1 = torch.zeros((2, 2)).cuda(), torch.zeros((2, 2)).cuda()

        for i in range(10):
            x, y = dataset.getImageAndLabel(city, i, torchformat=True)
            x, y = x.cuda(), y.cuda().float()

            h, w = y.shape[0], y.shape[1]
            globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
            power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
            x = power2resize(x)

            z = util.largeforward(net, x.unsqueeze(0))
            z = globalresize(z)
            p = z[0, 1, :, :] - z[0, 0, :, :]
            p = torch.nn.functional.sigmoid(p * 0.01)
            z = (z[0, 1, :, :] > z[0, 0, :, :]).float()

            border = util.getborder(y)
            border = 1 - border

            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                cm[a][b] += torch.sum((z == a).float() * (y == b).float())
                cm1[a][b] += torch.sum((z == a).float() * (y == b).float() * border)
            globalcm += cm
            globalcm1 += cm1

            if True:
                xpath, ypath = dataset.getPath(city, i)
                outradix = "build/" + city + str(i)
                digitanie.writeImage(xpath, x.cpu().numpy(), outradix + "_x.tif")
                digitanie.writeImage(xpath, y.cpu().numpy(), outradix + "_y.tif")
                digitanie.writeImage(xpath, z.cpu().numpy(), outradix + "_z.tif")
                digitanie.writeImage(xpath, p.cpu().numpy(), outradix + "_p.tif")

        print("perf0=", util.perf(cm))
        print("perf1=", util.perf(cm1))
        print("bords=", util.perf(cm - cm1))

print("global result")
print("perf0=", util.perf(globalcm))
print("perf1=", util.perf(globalcm1))
print("bords=", util.perf(globalcm - globalcm1))
