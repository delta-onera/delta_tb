import os
import sys
import torch

if len(sys.argv) == 1:
    print("no model selected")
    quit()

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

net = torch.load(sys.argv[1])
torch.save(net.state_dict(), "rahhhhwtftorchload.pth")
