import os
import torch
import torchvision
import dataloader
import PIL
import PIL.Image
import numpy

assert torch.cuda.is_available()

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/test/")


def crossentropy(y, z):
    class_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    tmp = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).cuda())
    return tmp(z, y.long())


def dicelossi(y, z, i):
    eps = 0.001
    z = z.softmax(dim=1)

    indexmap = torch.ones(z.shape).cuda()
    indexmap[:, i, :, :] = 0

    z0, z1 = z[:, i, :, :], (z * indexmap).sum(1)
    y0, y1 = (y == i).float(), (y != i).float()

    inter0, inter1 = (y0 * z0).sum(), (y1 * z1).sum()
    union0, union1 = (y0 + z1 * y0).sum(), (y1 + z0 * y1).sum()

    if union0 < eps or union1 < eps or union0 < inter0 or union1 < inter1:
        return 0

    iou0, iou1 = (inter0 + eps) / (union0 + eps), (inter1 + eps) / (union1 + eps)
    iou = 0.5 * (iou0 + iou1)
    return 1 - iou


def diceloss(y, z):
    alldice = torch.zeros(12).cuda()
    for i in range(12):
        alldice[i] = dicelossi(y, z, i)
    return alldice.mean()


print("test")


for i in range(len(dataset.paths)):
    if i % 100 == 99:
        print(i, "/", len(dataset.paths))

    x = dataset.getImageAndLabel(i)
    x = x.cuda()

    net = torch.load("../baseline/build/model.pth")
    net = net.cuda()
    net.eval()
    with torch.no_grad():
        feature = net.forwardbackbone(x.unsqueeze(0))

    with torch.no_grad():
        z = net.forwardhead(feature)
        v, py = z.max(1)
        py = (
            py * (v >= float(v.flatten().median())).float()
            + 12 * (v < float(v.flatten().median())).float()
        )

        # seuil = torch.zeros(12)
        # for
        # seuil, _ = v.flatten().sort()
        # seuil = seuil[-64] # 64 best pixel
        # best = py * (v>seuil).float()+12*(v<seuil).float()

    params = [
        net.final1.weight,
        net.final1.bias,
        net.final2.weight,
        net.final2.bias,
        net.final3.weight,
        net.final3.bias,
        net.classif.weight,
        net.classif.bias,
    ]
    optimizer = torch.optim.Adam(params, lr=0.000001)
    for j in range(160):
        feature.requires_grad_()
        z = net.forwardhead(feature)
        ce = crossentropy(py, z)
        dice = diceloss(py, z)

        loss = ce + dice

        if j % 40 == 0:
            print(j, loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

    with torch.no_grad():
        z = net.forwardhead(feature)
        _, z = z[0].max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        z = PIL.Image.fromarray(z)
        z.save("build/PRED_" + dataset.getName(i), compression="tiff_lzw")
