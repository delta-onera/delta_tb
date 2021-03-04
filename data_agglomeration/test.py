import os
import sys
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

print("load model")
if whereIam == "super":
    sys.path.append("/home/achanhon/github/segmentation_models/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/segmentation_models/pytorch-image-models")
    sys.path.append(
        "/home/achanhon/github/segmentation_models/pretrained-models.pytorch"
    )
    sys.path.append(
        "/home/achanhon/github/segmentation_models/segmentation_models.pytorch"
    )
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

import segmentation_models_pytorch

with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()


print("massif benchmark")
import dataloader

root, availabledata, weaklysupervised = dataloader.getindexeddata()


def accu(cm):
    return 100.0 * (cm[name][0][0] + cm[name][1][1]) / np.sum(cm[name])


def f1(cm):
    return 50.0 * cm[name][0][0] / (
        cm[name][0][0] + cm[name][1][0] + cm[name][0][1]
    ) + 50.0 * cm[name][1][1] / (cm[name][1][1] + cm[name][1][0] + cm[name][0][1])


cm = {}
with torch.no_grad():
    for name in availabledata:
        print("loading", name)
        data = dataloader.SegSemDataset(root + name + "/test/")

        cm[name] = np.zeros((2, 2), dtype=int)
        for i in range(data.nbImages):
            imageraw, label = data.getImageAndLabel(i)

            image = torch.Tensor(np.transpose(imageraw, axes=(2, 0, 1))).unsqueeze(0)
            globalresize = torch.nn.AdaptiveAvgPool2d((image.shape[2], image.shape[3]))
            power2resize = torch.nn.AdaptiveAvgPool2d(
                ((image.shape[2] // 32) * 32, (image.shape[3] // 32) * 32)
            )
            image = power2resize(image)

            if image.shape[2] < 512 and image.shape[3] < 512:
                pred = net(image.to(device))
            else:
                pred = dataloader.largeforward(net, image, device)

            pred = globalresize(pred)
            _, pred = torch.max(pred[0], 0)
            pred = pred.cpu().numpy()

            assert label.shape == pred.shape

            ### for weakly labelled area
            ### we do not want 1 in 0 area
            ### but we may have 0 in 1 area
            ### assuming they will be some 1 in 1 area
            ### we offer to considered label*pred as label
            if name in weaklysupervised:
                label = label * pred

            cm[name] += confusion_matrix(label.flatten(), pred.flatten(), labels=[0, 1])

            if name in ["toulouse", "potsdam"]:
                imageraw = PIL.Image.fromarray(np.uint8(imageraw))
                imageraw.save("build/" + name + "_" + str(i) + "_x.png")
                label = PIL.Image.fromarray(np.uint8(label) * 255)
                label.save("build/" + name + "_" + str(i) + "_y.png")
                pred = PIL.Image.fromarray(np.uint8(pred) * 255)
                pred.save("build/" + name + "_" + str(i) + "_z.png")

        print(cm[name][0][0], cm[name][0][1], cm[name][1][0], cm[name][1][1])
        print(
            accu(cm),
            f1(cm),
        )

print("supervised results")
for name in availabledata:
    if name not in weaklysupervised:
        print(name, f1(cm))

print("weakly supervised results -- performance may be dramatically biased")
for name in availabledata:
    if name in weaklysupervised:
        print(name, f1(cm))
