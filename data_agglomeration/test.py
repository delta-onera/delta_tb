import sys
import os
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

whereIam = os.uname()[1]
print(sys.argv)
assert len(sys.argv) > 1


print("load model")
if whereIam in ["super", "ldtis706z"]:
    sys.path.append("/home/achanhon/github/segmentation_models/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/segmentation_models/pytorch-image-models")
    sys.path.append(
        "/home/achanhon/github/segmentation_models/pretrained-models.pytorch"
    )
    sys.path.append(
        "/home/achanhon/github/segmentation_models/segmentation_models.pytorch"
    )
if whereIam == "wdtis719z":
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.to(device)


print("load data")
if whereIam in ["super", "wdtis719z"]:
    root = "/data/"
if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"

assert sys.argv[1] in ["ISPRS", "SEMITY", "AIRS", "DFC2015", "INRIA", "MINI"]
if len(sys.argv) > 1:
    flag = sys.argv[2]
else:
    flag = "TR"

import segsemdata
if sys.argv[1] == "ISPRS":
    data = segsemdata.makeISPRS(datasetpath=root + "ISPRS_POTSDAM", flag=flag)
if sys.argv[1] == "DFC2015":
    data = segsemdata.makeDFC2015(datasetpath=root + "DFC2015", flag=flag)
if sys.argv[1] == "SEMITY":
    data = segsemdata.makeSEMCITY(datasetpath=root + "SEMCITY_TOULOUSE", flag=flag)
if sys.argv[1] == "AIRS":
    data = segsemdata.makeAIRSdataset(datasetpath=root + "AIRS", flag=flag)



print("test")
import torch.nn as nn


def tileforward(net, data, tilesize=128):
    if (
        128 <= data.shape[2] <= 512
        and data.shape[2] % 32 == 0
        and 128 <= data.shape[3] <= 512
        and data.shape[3] % 32 == 0
    ):
        return net(data)

    if data.shape[2] <= 512 and data.shape[3] <= 512:
        globalresize = nn.AdaptiveAvgPool2d((data.shape[2], data.shape[3]))
        power2resize = nn.AdaptiveAvgPool2d(
            (max(128, (data.shape[2] // 32) * 32), max(128, (data.shape[3] // 32) * 32))
        )

        data = power2resize(data)
        data = net(data)
        data = globalresize(data)
        return data

    if net.training or data.shape[0] != 1:
        print(
            "it is impossible to train on too large tile or to do the inference on a large batch of large images"
        )
        quit()

    with torch.no_grad():
        device = data.device
        globalresize = nn.AdaptiveAvgPool2d((data.shape[2], data.shape[3]))
        power2resize = nn.AdaptiveAvgPool2d(
            (max(128, (data.shape[2] // 32) * 32), max(128, (data.shape[3] // 32) * 32))
        )

        data = power2resize(data)

        output = torch.zeros(1, 2, data.shape[2], data.shape[3]).cpu()
        for row in range(0, data.shape[2] - tilesize + 1, 32):
            for col in range(0, data.shape[3] - tilesize + 1, 32):
                output[:, :, row : row + tilesize, col : col + tilesize] += net(
                    data[:, :, row : row + tilesize, col : col + tilesize]
                ).cpu()

        return globalresize(output.to(device))


with torch.no_grad():
    net.eval()
    if True:
        nbclasses = len(data.setofcolors)
        cm = np.zeros((nbclasses, nbclasses), dtype=int)
        for name in data.getnames():
            image, label = data.getImageAndLabel(name, innumpy=False)
            pred = tileforward(net, image.to(device))
            _, pred = torch.max(pred[0], 0)
            pred = pred.cpu().numpy()

            assert label.shape == pred.shape

            cm += confusion_matrix(
                label.flatten(), pred.flatten(), list(range(nbclasses))
            )

            pred = PIL.Image.fromarray(data.vtTOcolorvt(pred))
            pred.save("build/" + name + "_z.png")

        print(data.datasetname)
        print(segsemdata.getstat(cm))
        print(cm)
