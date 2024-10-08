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

outputname = "build/model.pth"
if len(sys.argv) > 1:
    outputname = sys.argv[1]
os.system("cat train3.py")

whereIam = os.uname()[1]

print("define model")
import collections
import random
import lipschitz_unet

net = lipschitz_unet.UNET(debug=False)
net = net.cuda()
net.train()


print("load data")
import dataloader

miniworld = dataloader.MiniWorld(flag="custom", custom=["potsdam/train"])

earlystopping = miniworld.getrandomtiles(5000, 128, 32)
weights = torch.Tensor([1, miniworld.balance, 0.00001]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

print("train")


def accu(cm):
    return 100.0 * (cm[0][0] + cm[1][1]) / np.sum(cm)


def f1(cm):
    return 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1]) + 50.0 * cm[1][1] / (
        cm[1][1] + cm[1][0] + cm[0][1]
    )


def trainaccuracy():
    cm = np.zeros((3, 3), dtype=int)
    net.eval()
    with torch.no_grad():
        for inputs, targets in earlystopping:
            inputs, targets = inputs.to(device), targets.to(device)

            targets = dataloader.convertIn3class(targets)

            outputs = net(inputs)
            _, pred = outputs.max(1)
            for i in range(pred.shape[0]):
                cm += confusion_matrix(
                    pred[i].cpu().numpy().flatten(),
                    targets[i].cpu().numpy().flatten(),
                    labels=[0, 1, 2],
                )
    return cm[0:2, 0:2]


optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=400)
meanreg = collections.deque(maxlen=400)
nbepoch = 100
batchsize = 8

for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)

    XY = miniworld.getrandomtiles(5000, 128, batchsize)
    for x, y in XY:
        x, y = x.to(device), y.to(device)

        preds = net(x)
        tmp = torch.zeros(preds.shape[0], 1, preds.shape[2], preds.shape[3])
        tmp = tmp.to(device)
        preds = torch.cat([preds, tmp], dim=1)

        yy = dataloader.convertIn3class(y)

        loss = criterion(preds, yy)
        reg = net.getLipschitzbound()

        meanloss.append(loss.cpu().data.numpy())
        meanreg.append(reg.cpu().data.numpy())

        if epoch > 5:
            loss = loss * 0.5
        if epoch > 10:
            loss = loss * 0.5
        if epoch > 40:
            loss = loss * 0.1
        if epoch > 80:
            loss = loss * 0.1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
        optimizer.step()

        if random.randint(0, 30) == 0:
            print(
                "loss=",
                (sum(meanloss) / len(meanloss)),
                "reg=",
                (sum(meanreg) / len(meanreg)),
            )

    print("backup model")
    torch.save(net, outputname)
    cm = trainaccuracy()
    print("accuracy,iou", accu(cm), f1(cm))
    if accu(cm) > 92:
        print(":-)")
        quit()

print("training stops after reaching time limit")
