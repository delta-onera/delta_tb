import os
import sys

if len(sys.argv) > 1:
    outputname = sys.argv[1]
else:
    outputname = "build/model.pth"

whereIam = os.uname()[1]
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")
else:
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")

import torch
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp

print("define model")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()
net = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)
net = net.cuda()
net.train()


print("load data")
import dataloader

miniworld = dataloader.MiniWorld()

print("train")
import collections
import random

earlystopping = miniworld.getrandomtiles(5000, 128, 32)


def trainaccuracy():
    with torch.no_grad():
        net.eval()
        good, tot = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        for x, y in earlystopping:
            x, y = x.cuda(), y.cuda()

            D = dataloader.distanceToBorder(y)
            tot += torch.sum(D)

            z = net(x)
            _, z = z.max(1)

            cool = (z == y).float()
            cool *= D
            good += torch.sum(cool)
        return 100.0 * good.cpu().numpy() / tot.cpu().numpy()


# weights = torch.Tensor([1, miniworld.balance]).cuda()
# criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
criterionbis = smp.losses.dice.DiceLoss(mode="multiclass")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
nbepoch = 800
batchsize = 16

for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = miniworld.getrandomtiles(10000, 128, batchsize)
    for x, y in XY:
        x, y = x.cuda(), y.cuda()
        D = dataloader.distanceToBorder(y)

        z = net(x)

        if random.randint(0, 10) == 0:
            weights = torch.Tensor([1, miniworld.balance]).cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
        else:
            nb0, nb1 = torch.sum((y == 0).float()), torch.sum((y == 1).float())
            weights = torch.Tensor([1, nb0 / nb1]).cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")

        CE = criterion(z, y)
        CE = torch.mean(CE * D)
        dice = criterionbis(z, y)
        loss = CE + dice

        meanloss.append(loss.cpu().data.numpy())

        if epoch > 30:
            loss = loss * 0.5
        if epoch > 90:
            loss = loss * 0.5
        if epoch > 160:
            loss = loss * 0.5
        if epoch > 400:
            loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
        optimizer.step()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    print("backup model")
    torch.save(net, outputname)
    accu = trainaccuracy()
    print("accuracy", accu)

    if accu > 98:
        print("training stops after reaching high training accuracy")
        quit()
print("training stops after reaching time limit")
