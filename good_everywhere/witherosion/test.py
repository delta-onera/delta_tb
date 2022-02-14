import os
import sys
import numpy
import PIL
from PIL import Image
import torch
import torch.backends.cudnn as cudnn

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
import dataloader

print("load data")
miniworld = dataloader.DigitanieALL()

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


def erosion(z, size=2):
    zz = -torch.nn.functional.max_pool2d(
        -z, kernel_size=2 * size + 1, stride=1, padding=size
    )
    zzz = z.clone()
    zzz[:, size:-size, size:-size] = zz[:, size:-size, size:-size]
    return zzz


cm = torch.zeros((len(miniworld.cities), 2, 2)).cuda()
with torch.no_grad():
    for k, city in enumerate(miniworld.cities):
        print(k, city)

        for i in range(miniworld.NB[city]):
            x, y = miniworld.getImageAndLabel(city, i, torchformat=True)
            x, y = x.cuda(), y.cuda()

            h, w = y.shape[0], y.shape[1]
            D = dataloader.distancetransform(y)
            globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
            power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
            x = power2resize(x)

            z = largeforward(net, x.unsqueeze(0))
            z = erosion(z[:, 1, :, :] - z[:, 0, :, :])
            z = globalresize(z)
            z = (z[0] > 0).float()

            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                cm[k][a][b] = torch.sum((z == a).float() * (y == b).float() * D)

            if True:
                nextI = len(os.listdir("build"))
                debug = dataloader.torchTOpil(globalresize(x))
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_x.png")
                debug = (2.0 * y - 1) * D * 127 + 127
                debug = debug.cpu().numpy()
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_y.png")
                debug = z.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_z.png")

        print("perf=", dataloader.perf(cm[k]))
        numpy.savetxt("build/tmp.txt", dataloader.perf(cm).cpu().numpy())

print("-------- results ----------")
for k, city in enumerate(miniworld.cities):
    print(city, dataloader.perf(cm[k]))

cm = torch.sum(cm, dim=0)
print("miniworld", dataloader.perf(cm))
