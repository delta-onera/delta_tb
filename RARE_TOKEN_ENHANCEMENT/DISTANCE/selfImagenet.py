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
w = torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
dataset = torchvision.datasets.ImageFolder(
    "/scratchf/IMAGENET/train", transform=w.transforms()
)
dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=True, batch_size=16, num_workers=2
)

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
i = 0
for epoch in range(3):
    print(epoch)
    for x, _ in dataloader:
        x = x.cuda()
        z = [net.embedding(x[i]) for i in range(x.shape[0])]
        auxloss = [common.getRarityLoss(z[i]) for i in range(x.shape[0])]
        auxloss = sum(auxloss) / len(auxloss)

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

        i += 1
        if i == 10000:
            break
