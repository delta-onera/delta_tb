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
losslog = torch.zeros(50).cuda()
for epoch in range(20):
    for i, (x, y) in enumerate(trainloader):
        x, y = x.cuda(), y.cuda()
        pred = net(x)
        classifloss = lossfunction(pred, y)

        loss = classifloss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        with torch.no_grad():
            losslog[i % 50] = classifloss.clone()
            if i % 50 == 49:
                print(float(losslog.mean(0)))
                losslog = torch.zeros(50).cuda()

    torch.save(net, "build/model.pth")
