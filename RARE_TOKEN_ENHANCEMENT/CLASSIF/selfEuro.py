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

trainset = common.EurosatSplit("train")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
unlabeled = common.getEurosat()
unlabeledloader = torch.utils.data.DataLoader(unlabeled, batch_size=64, shuffle=True)

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
lossfunction = torch.nn.CrossEntropyLoss()
losslog = torch.zeros(50, 2).cuda()
for epoch in range(20):
    for i, ((x, y), (z, _)) in enumerate(zip(trainloader, unlabeledloader)):
        x, y, z = x.cuda(), y.cuda(), z.cuda()
        pred = net(x)
        classifloss = lossfunction(pred, y)

        # self loss: rare token enhancement
        feat = net.embedding(z)
        tmp = [common.getRarityLoss(feat[i]) for i in range(z.shape[0])]
        rarityloss = sum(tmp) / len(tmp)

        loss = classifloss + rarityloss * 0.1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        with torch.no_grad():
            losslog[i % 50][0] = classifloss.clone()
            losslog[i % 50][1] = rarityloss.clone()
            if i % 50 == 49:
                losslog = losslog.mean(0)
                a, b = float(losslog[0]), float(losslog[1])
                print(a, b)
                losslog = torch.zeros(50, 2).cuda()

    torch.save(net, "build/model.pth")
