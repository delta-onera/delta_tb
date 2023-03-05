import os
import torch
import torchvision
import dataloader
import sys

assert torch.cuda.is_available()

assert len(sys.argv) > 1
model = sys.argv[1]

print("load model")
with torch.no_grad():
    net = torch.load("../featuretrain/build/" + model + ".pth")
    net = net.cuda()
    net.eval()

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/train/", "1/4")

print("test")

max1, max2, nblower0, tot = 0, 0, 0, 0
with torch.no_grad():
    for i in range(len(dataset.paths)):
        if i % 100 == 99:
            print(i, "/", len(dataset.paths))
        x, _ = dataset.getImageAndLabel(i)
        x = x.cuda()

        z = net(x.unsqueeze(0))[0]
        V, I = z.max(0)
        nblower0 = nblower0 + (V <= 0).flatten().float().sum()
        tot = tot + x.flatten().shape[0]

        z = torch.nn.functional.leaky_relu(z)
        torch.save(z, "build/" + model + "/" + dataset.getName(i) + ".pth")

        max1 = max(max1, V.flatten().max())
        z[I] = 0
        V, _ = z.max(0)
        max2 = max(max2, V.flatten().max())

print("##############################")
print(max1, max2, nblower0, tot)
os._exit(0)
