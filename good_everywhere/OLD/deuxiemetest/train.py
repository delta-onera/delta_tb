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
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import cropextractor
import dataloader

print("load data")
miniworld = dataloader.MiniWorld(flag="train")

print("define model")
net = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
net = net.cuda()
net.train()


print("train")
if len(sys.argv) == 2 and sys.argv[1] == "penalizemin":
    print("mode penalize min")
else:
    print("mode baseline")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
if whereIam in ["ldtis706z", "wdtim719z"]:
    batchsize = 16
else:
    batchsize = 32
nbbatchs = 200000
printloss = torch.zeros(1).cuda()
stats = torch.zeros((len(miniworld.cities), 2, 2)).cuda()
worse = set([i for i in range(stats.shape[0])])
miniworld.start()

for i in range(nbbatchs):
    x, y, batchchoise = miniworld.getbatch(batchsize)
    x, y = x.cuda(), y.cuda()
    z = net(x)

    D = dataloader.distancetransform(y)
    nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
    weights = torch.Tensor([1, nb0 / (nb1 + 1)]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
    CE = criterion(z, y.long())
    CE = CE * D
    CE = torch.mean(CE, dim=1)
    CE = torch.mean(CE, dim=1)

    assert CE.shape[0] == len(batchchoise)
    for j in range(CE.shape[0]):
        if batchchoise[j] in worse:
            CE[j] *= 6.0
    CE = torch.mean(CE)

    criteriondice = smp.losses.dice.DiceLoss(mode="multiclass")
    dice = criteriondice(z, y.long())
    loss = CE + 0.1 * dice

    with torch.no_grad():
        printloss += loss.clone().detach()
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

    with torch.no_grad():
        z = (z[:, 1, :, :] > z[:, 0, :, :]).float()
        for j in range(batchsize):
            cm = torch.zeros(2, 2).cuda()
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                cm[a][b] = torch.sum((z[j] == a).float() * (y[j] == b).float() * D[j])
            stats[batchchoise[j]] += cm

    if i % 100 == 99:
        print(i, "/200000", printloss / 100)
        printloss = torch.zeros(1).cuda()

    if i % 1000 == 999:
        torch.save(net, "build/model.pth")
        cm = torch.sum(stats, dim=0)
        perf = dataloader.perf(cm)
        print("perf", perf)
        if perf[0] > 92:
            print("training stops after reaching high training accuracy")
            os._exit(0)
        else:
            if len(sys.argv) == 2 and sys.argv[1] == "penalizemin":
                perfs = [dataloader.perf(stats[j])[0] for j in range(stats.shape[0])]
                tmp = perfs[:]
                sorted(tmp)
                threshold = tmp[stats.shape[0] // 3 + 1]
                worse = set([j for j in range(stats.shape[0]) if perfs[j] <= threshold])
                print("penalty increased for", worse)

            stats = torch.zeros((len(miniworld.cities), 2, 2)).cuda()

print("training stops after reaching time limit")
os._exit(0)
