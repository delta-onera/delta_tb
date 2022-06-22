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
miniworld = digitanie.DigitanieALL()

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


cm = torch.zeros((len(miniworld.cities), 2, 2)).cuda()
with torch.no_grad():
    for k, city in enumerate(miniworld.cities):
        print(k, city)

        for i in range(miniworld.NB[city]):
            x, y = miniworld.getImageAndLabel(city, i, torchformat=True)
            x, y = x.cuda(), y.cuda()

            h, w = y.shape[0], y.shape[1]
            D = digitanie.distancetransform(y)
            globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
            power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
            x = power2resize(x)

            z = largeforward(net, x.unsqueeze(0))
            z = globalresize(z)
            z = erosion(z, size=int(sys.argv[1]))
            p = z[0, 1, :, :] - z[0, 0, :, :]
            p = torch.nn.functional.sigmoid(p * 50)
            z = (z[0, 1, :, :] > z[0, 0, :, :]).float()

            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                cm[k][a][b] = torch.sum((z == a).float() * (y == b).float() * D)

            if True:
                xpath, ypath = miniworld.getPath(city, i)
                outradix = "build/" + city + str(i)
                digitanie.writeImage(xpath, x.cpu().numpy(), outradix + "_x.tif")
                digitanie.writeImage(ypath, y.cpu().numpy(), outradix + "_y.tif")
                digitanie.writeImage(ypath, z.cpu().numpy(), outradix + "_z.tif")
                digitanie.writeImage(ypath, p.cpu().numpy(), outradix + "_p.tif")

        print("perf=", digitanie.perf(cm[k]))
        print(cm[k])
        numpy.savetxt("build/tmp.txt", util.perf(cm).cpu().numpy())

print("-------- summary ----------")
for k, city in enumerate(miniworld.cities):
    print(city, digitanie.perf(cm[k]))

cm = torch.sum(cm, dim=0)
print("digitanie", digitanie.perf(cm))
