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
    net = torch.load("../../multiple/build/" + model + ".pth")
    net = net.cuda()
    net.eval()

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/test/", "1")

print("test")

with torch.no_grad():
    for i in range(len(dataset.paths)):
        if i % 100 == 99:
            print(i, "/", len(dataset.paths))
        x, _ = dataset.getImageAndLabel(i)
        x = x.cuda()

        z = net(x.unsqueeze(0))[0]

        z = torch.nn.functional.leaky_relu(z)
        torch.save(z, "build/" + model + "/" + dataset.getName(i) + ".pth")

os._exit(0)
