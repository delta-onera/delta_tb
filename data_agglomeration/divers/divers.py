import os
import sys
import torch

whereIam = os.uname()[1]

print("define model")
if whereIam == "super":
    sys.path.append("/home/achanhon/github/segmentation_models/EfficientNet-PyTorch")
    sys.path.append("/home/achanhon/github/segmentation_models/pytorch-image-models")
    sys.path.append(
        "/home/achanhon/github/segmentation_models/pretrained-models.pytorch"
    )
    sys.path.append(
        "/home/achanhon/github/segmentation_models/segmentation_models.pytorch"
    )
if whereIam == "wdtim719z":
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append(
        "/d/achanhon/github/EfficientNet-PyTorch"
    )  # https://github.com/lukemelas/EfficientNet-PyTorch
    sys.path.append(
        "/d/achanhon/github/pytorch-image-models"
    )  # https://github.com/rwightman/pytorch-image-models
    sys.path.append(
        "/d/achanhon/github/pretrained-models.pytorch"
    )  # https://github.com/Cadene/pretrained-models.pytorch
    sys.path.append(
        "/d/achanhon/github/segmentation_models.pytorch"
    )  # https://github.com/qubvel/segmentation_models.pytorch

import segmentation_models_pytorch as smp


def wtf(*features):
    c2, c3, c4, c5 = features[-4:]
    print(c2.shape)
    print(c3.shape)
    print(c4.shape)
    print(c5.shape)


net = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
with torch.no_grad():
    net.eval()

    x = torch.zeros(1, 3, 128, 128)
    feature = net.encoder(x)
    print("##############################")
    for z in feature:
        print(z.shape)

    print("##############################")
    wtf(*feature)

    feature = net.decoder(*feature)
    print(feature.shape)

net = smp.FPN(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
net.head = smp.base.SegmentationHead(16, 2, kernel_size=1)
with torch.no_grad():
    net.eval()

    x = torch.zeros(1, 3, 128, 128)
    feature = net.encoder(x)
    print("##############################")
    for z in feature:
        print(z.shape)

    print("##############################")

    wtf(*feature)

    feature = net.decoder(*feature)
    print(feature.shape)

net = smp.FPN(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
net.head = smp.base.SegmentationHead(16, 2, kernel_size=1)
with torch.no_grad():
    net.eval()

    x = torch.zeros(1, 3, 128, 128)
    feature = net.encoder(x)
    print("##############################")
    for z in feature:
        print(z.shape)

    print("##############################")

    wtf(*feature)

    feature = net.decoder(*feature)
    print(feature.shape)
