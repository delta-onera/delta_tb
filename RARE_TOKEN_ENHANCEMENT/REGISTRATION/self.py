import sys

print(sys.argv)
if len(sys.argv) > 1:
    modelname = sys.argv[1]
    assert modelname in ["EfficientNet", "EfficientNetV2"]
else:
    modelname = "EfficientNet"
print("modelname", modelname)

import torch
import torchvision
import common

print("load data")
trainset = common.S2Looking("train")
unlabelised = common.S2Looking("all")

print("load encoder")
if modelname == "EfficientNet":
    encoder = common.getEfficientNet()
if modelname == "EfficientNetV2":
    encoder = common.getEfficientNetV2()
net = common.MyNet(encoder)
net = net.cuda()
net.eval()

print("set train setting")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
losslog = torch.zeros(200, 2).cuda()
for i in range(7000):
    img1, img2, cm = trainset.getrand()
    img1, img2, cm = img1.cuda(), img2.cuda(), cm.cuda()
    pred = net(img1, img2)
    mainloss = ((cm - pred) ** 2).sum()

    # self loss: rare token enhancement
    img1, img2, _ = unlabelised.getrand()
    feat1 = net.embedding(img1.cuda())
    feat2 = net.embedding(img2.cuda())
    auxloss1 = common.getRarityLoss(feat1.flatten(1))
    auxloss2 = common.getRarityLoss(feat2.flatten(1))
    auxloss = (auxloss1 + auxloss2) / 2

    loss = mainloss * 200 + auxloss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    optimizer.step()

    with torch.no_grad():
        losslog[i % 200][0] = mainloss.clone()
        losslog[i % 200][1] = auxloss.clone()
        if i % 200 == 199:
            losslog = losslog.mean(0)
            a, b = float(losslog[0].mean()), float(losslog[1].mean())
            print(a, b)
            losslog = torch.zeros(200, 2).cuda()

torch.save(net, "build/model.pth")
