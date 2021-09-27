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
    accu = 100.0 * (cm[0][0] + cm[1][1]) / numpy.sum(cm)
    iou = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1]) + 50.0 * cm[1][1] / (
        cm[1][1] + cm[1][0] + cm[0][1]
    )
    return iou, accu


cmforlogging = []
cm = {}
with torch.no_grad():
    for town in miniworld.towns:
        print(town)
        cm[town] = torch.zeros((2, 2)).cuda()
        for i in range(miniworld.data[town].nbImages):
            imageraw, label = miniworld.data[town].getImageAndLabel(i)

            label = torch.Tensor(label).cuda()
            distance = dataloader.distancetransform(label)

            image = torch.Tensor(numpy.transpose(imageraw, axes=(2, 0, 1))).unsqueeze(0)
            h, w = image.shape[2], image.shape[3]
            globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
            power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
            image = power2resize(image)

            pred = dataloader.largeforward(net, image)

            pred = globalresize(pred)
            _, pred = torch.max(pred[0], 0)

            cm[town][0][0] += torch.sum(
                (pred == 0).float() * (label == 0).float() * distance
            )
            cm[town][1][1] += torch.sum(
                (pred == 1).float() * (label == 1).float() * distance
            )
            cm[town][1][0] += torch.sum(
                (pred == 1).float() * (label == 0).float() * distance
            )
            cm[town][0][1] += torch.sum(
                (pred == 0).float() * (label == 1).float() * distance
            )

            if town in ["potsdam/test", "chicago/test", "Austin/test"]:
                nextI = len(os.listdir("build"))
                debug = image[0].cpu().numpy()
                debug = numpy.transpose(debug, axes=(1, 2, 0))
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_x.png")
                debug = (2.0 * label - 1) * distance * 127 + 127
                debug = debug.cpu().numpy()
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_y.png")
                debug = pred.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + str(nextI) + "_z.png")

        cm[town] = cm[town].cpu().numpy()
        iou, accu = perf(cm[town])
        print("iou=", iou)
        cmforlogging.append(iou)
        debug = numpy.asarray(cmforlogging)
        numpy.savetxt("build/logtest.txt", debug)

print("-------- results ----------")
for town in miniworld.towns:
    print(town, perf(cm[town]))

globalcm = numpy.zeros((2, 2))
for town in miniworld.towns:
    globalcm += cm[town]
print("miniworld", perf(globalcm))
