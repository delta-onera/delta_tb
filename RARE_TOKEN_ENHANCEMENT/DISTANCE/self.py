import sys

print(sys.argv)
if len(sys.argv) > 1:
    modelname = sys.argv[1]
    assert modelname in ["EfficientNet", "EfficientNetV2"]
else:
    modelname = "EfficientNet"
print("modelname", modelname)

if len(sys.argv) > 2:
    dataname = sys.argv[2]
    assert dataname in ["S2L", "LEVIR", "OSCD"]
else:
    dataname = "EfficientNet"
print("dataname", dataname)


import torch
import torchvision
import common

print("load data")
if dataname == "S2L":
    unlabelised = common.S2Looking("all")
if dataname == "LEVIR":
    unlabelised = common.LEVIR("all")
if dataname == "OSCD":
    unlabelised = common.OSCD("all")

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
losslog = torch.zeros(200).cuda()
for i in range(10000):
    img1, img2, _ = unlabelised.getrand()
    feat1 = net.embedding(img1.cuda())
    feat2 = net.embedding(img2.cuda())
    auxloss1 = common.getRarityLoss(feat1.flatten(1))
    auxloss2 = common.getRarityLoss(feat2.flatten(1))
    auxloss = (auxloss1 + auxloss2) / 2

    loss = auxloss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    optimizer.step()

    with torch.no_grad():
        losslog[i % 200] = auxloss.clone()
        if i % 200 == 199:
            print(float(losslog.mean(0)))
            losslog = torch.zeros(200).cuda()
