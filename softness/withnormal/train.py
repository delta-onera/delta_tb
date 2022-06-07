import os
import sys

import numpy
import PIL
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torchvision

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()

sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
sys.path.append("/d/achanhon/github/pytorch-image-models")
sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import noisyairs

print("load data")
dataset = noisyairs.AIRS("/train/")

print("define model")
net = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4,
)
net = net.cuda()
net.train()


print("train")
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
printloss = torch.zeros(3).cuda()
stats = torch.zeros((2, 2)).cuda()
batchsize = 32
nbbatchs = 75000
dataset.start()


def diceloss(y, z, D):
    eps = 0.00001
    z = z.log_softmax(dim=1).exp()
    z0, z1 = z[:, 0, :, :], z[:, 1, :, :]
    y0, y1 = (y == 0).float(), (y == 1).float()

    inter0, inter1 = (y0 * z0 * D).sum(), (y1 * z1 * D).sum()
    union0, union1 = ((y0 + z1 * y0) * D).sum(), ((y1 + z0 * y1) * D).sum()
    iou0, iou1 = (inter0 + eps) / (union0 + eps), (inter1 + eps) / (union1 + eps)
    iou = 0.5 * (iou0 + iou1)

    return 1 - iou


def selfcoherence(z, predtH, predtW):
    h, w = z.shape[-2], z.shape[-1]
    tmp = z[:, 1, :, :] - z[:, 0, :, :]
    predborder = noisyairs.isborder((tmp > 0).float(), 1)

    l = [(0, 1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]
    l = [(i + 1, j + 1) for i, j in l]
    dz = torch.zeros(8, z.shape[0], h + 3, w + 3).cuda()
    for k, (i, j) in enumerate(l):
        dz[k, :, i : i + h, j : j + w] = tmp

    gradW = (dz[0] - dz[1] + dz[4] - dz[2] + dz[-1] - dz[-3]) / 3
    gradH = (dz[-1] - dz[4] + dz[-2] - dz[3] + dz[-3] - dz[-2]) / 3
    gradH, gradW = gradH[:, 1 : 1 + h, 1 : 1 + w], gradW[:, 1 : 1 + h, 1 : 1 + w]

    WH = torch.sqrt(gradW * gradW + gradH * gradH + 0.001)
    gradW = gradW / WH
    gradH = gradH / WH

    normal_tangente = gradH * predtH + gradW * predtW
    return torch.mean(normal_tangente.abs() * predborder)


for i in range(nbbatchs):
    x, y, tangent = dataset.getBatch(batchsize)
    x, y, tangent = x.cuda(), y.cuda(), tangent.cuda()
    z = net(x)

    pixelwithtangent = (tangent[:, 0, :, :] != 0).int().cuda()
    tangentH, tangentW = tangent[:, 1, :, :], tangent[:, 2, :, :]
    tangentH, tangentW = tangentH / 127 - 1, tangentW / 127 - 1
    D = 1 - pixelwithtangent

    pred = z[:, 0:2, :, :]
    CE = criterion(pred, y)
    CE = torch.mean(CE * D)
    dice = diceloss(y, pred, D)
    segloss = CE + dice

    predtH, predtW = z[:, 2, :, :], z[:, 3, :, :]
    WH = torch.sqrt(predtH * predtH + predtW * predtW + 0.001)
    predtW = predtW / WH
    predtH = predtH / WH

    regloss = (tangentH - predtH).abs() + (tangentW - predtW).abs()
    reglossbis = (tangentH + predtH).abs() + (tangentW + predtW).abs()
    reglossfinal = torch.minimum(regloss, reglossbis * 1.3)
    reglossfinal = torch.mean(reglossfinal * pixelwithtangent)

    loss = segloss + 2 * reglossfinal + 2 * selfcoherence(pred, predtH, predtW)

    with torch.no_grad():
        printloss[0] += segloss.clone().detach()
        printloss[1] += reglossfinal.clone().detach()
        printloss[2] += selfcoherence.clone().detach()
        z = (z[:, 1, :, :] > z[:, 0, :, :]).clone().detach().float()
        for j in range(batchsize):
            stats += noisyairs.confusion(y[j], z[j], size=1)

        if i < 10:
            print(i, "/", nbbatchs, printloss)
        if i < 1000 and i % 100 == 99:
            print(i, "/", nbbatchs, printloss / 100)
            printloss = torch.zeros(3).cuda()
        if i >= 1000 and i % 300 == 299:
            print(i, "/", nbbatchs, printloss / 300)
            printloss = torch.zeros(3).cuda()

        if i % 1000 == 999:
            torch.save(net, "build/model.pth")
            perf = noisyairs.perf(stats)
            print(i, "perf", perf)
            stats = torch.zeros((2, 2)).cuda()

    if i > nbbatchs * 0.1:
        loss = loss * 0.5
    if i > nbbatchs * 0.2:
        loss = loss * 0.5
    if i > nbbatchs * 0.5:
        loss = loss * 0.5
    if i > nbbatchs * 0.8:
        loss = loss * 0.5

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
    optimizer.step()

print("training stops after reaching time limit")
os._exit(0)
