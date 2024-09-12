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

print("load encoder")
if modelname == "EfficientNet":
    encoder = common.getEfficientNet()
if modelname == "EfficientNetV2":
    encoder = common.getEfficientNetV2()
net = common.MyNet(encoder)
net = net.cuda()
net.eval()

print("set train setting")
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
cefunction = torch.nn.CrossEntropyLoss()
losslog = torch.zeros(50, 2).cuda()
for i in range(trainset.NBITER):
    img1, img2, cm = trainset.getrand()
    img1, img2, cm = img1.cuda(), img2.cuda(), cm.cuda()
    pred = net(img1, img2)
    dice = common.dice(cm.float(), pred)
    cm = cm.flatten()
    pred = pred.flatten(1).transpose(0, 1)
    ce = cefunction(pred, cm)

    loss = 0.5 * ce + dice
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    optimizer.step()

    with torch.no_grad():
        losslog[i % 50][0] = ce.clone()
        losslog[i % 50][1] = dice.clone()
        if i % 50 == 49:
            losslog = losslog.mean(0)
            a, b = float(losslog[0]), float(losslog[1])
            print(a, b)
            losslog = torch.zeros(50, 2).cuda()

torch.save(net, "build/model.pth")
