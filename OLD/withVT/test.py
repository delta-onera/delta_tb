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
root = "/media/achanhon/bigdata/data/miniworld/christchurch/test/"
if len(sys.argv) == 1 or "vt" in sys.argv[1]:
    airs = dataloader.SegSemDataset(root, FLAGinteractif=0)
else:
    airs = dataloader.SegSemDataset(root, FLAGinteractif=100)
dataloader = airs.getFrozenTiles()

print("test")
import numpy
import PIL
from PIL import Image

cm = torch.zeros((2, 2)).cuda()
with torch.no_grad():
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        h, w = y.shape[0], y.shape[1]
        D = dataloader.distancetransform(y)

        z = net(x)
        z = (z[:, 1, :, :] > z[:, 0, :, :]).float()

        if len(sys.argv) == 1 or "cut" in sys.argv[1]:
            z, y = z[:, 64:, :], y[:, 64:, :]

        cm[0][0] += torch.sum((z == 0).float() * (y == 0).float() * D)
        cm[1][1] += torch.sum((z == 1).float() * (y == 1).float() * D)
        cm[1][0] += torch.sum((z == 1).float() * (y == 0).float() * D)
        cm[0][1] += torch.sum((z == 0).float() * (y == 1).float() * D)

        if True:
            nextI = len(os.listdir("build"))
            debug = x.cpu().numpy()
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

    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    print(iou0 + iou1, accu)
