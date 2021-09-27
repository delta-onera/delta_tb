import os
import sys
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
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import dataloader

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("load data")
miniworld = dataloader.MiniWorld("test")

print("test")
import numpy
import PIL
from PIL import Image


def perf(cm):
    if len(cm.shape) == 2:
        accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
        iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
        iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
        return torch.Tensor((iou0 + iou1, accu))
    else:
        out = torch.zeros(cm.shape[0], 2)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        return out


cm = torch.zeros((len(miniworld.towns), 2, 2)).cuda()
with torch.no_grad():
    for k, town in enumerate(miniworld.towns):
        print(k, town)
        for i in range(miniworld.data[town].nbImages):
            imageraw, label = miniworld.data[town].getImageAndLabel(i)

            y = torch.Tensor(label).cuda()
            h, w = y.shape[0], y.shape[1]
            D = dataloader.distancetransform(y)

            x = torch.Tensor(numpy.transpose(imageraw, axes=(2, 0, 1))).unsqueeze(0)
            globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
            power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
            x = power2resize(x)

            z = dataloader.largeforward(net, x)

            z = globalresize(z)
            z = (z[0, 1, :, :] > z[0, 0, :, :]).float()

            cm[k][0][0] += torch.sum((z == 0).float() * (y == 0).float() * D)
            cm[k][1][1] += torch.sum((z == 1).float() * (y == 1).float() * D)
            cm[k][1][0] += torch.sum((z == 1).float() * (y == 0).float() * D)
            cm[k][0][1] += torch.sum((z == 0).float() * (y == 1).float() * D)

            if town in ["potsdam/test", "chicago/test", "Austin/test"]:
                nextI = len(os.listdir("build"))
                debug = globalresize(x)[0].cpu().numpy()
                debug = numpy.transpose(debug, axes=(1, 2, 0))
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_x.png")
                debug = (2.0 * y - 1) * D * 127 + 127
                debug = debug.cpu().numpy()
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_y.png")
                debug = z.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_z.png")

        print("perf=", perf(cm[k]))
        numpy.savetxt("build/logtest.txt", perf(cm).cpu().numpy())

print("-------- results ----------")
for k, town in enumerate(miniworld.towns):
    print(town, perf(cm[k]))

cm = torch.sum(cm, dim=0)
print("miniworld", perf(cm))
