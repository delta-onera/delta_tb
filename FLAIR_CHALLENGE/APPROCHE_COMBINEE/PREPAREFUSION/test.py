import os
import torch
import torchvision
import dataloader
import sys

assert torch.cuda.is_available()


print("load model")
assert len(sys.argv) >= 3
assert sys.argv[1] in ["RGB", "RIE", "IGE", "IEB"]
assert sys.argv[2] in ["train", "test"]

root = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/APPROCHE_COMBINEE/"
with torch.no_grad():
    net = torch.load(root + sys.argv[1] + "/build/model.pth")
    net = net.cuda()
    net.eval()

print("load data")
dataset = dataloader.FLAIRTEST(
    "/scratchf/CHALLENGE_IGN/" + sys.argv[2] + "/", net.channels
)


print("test")

with torch.no_grad():
    for i in range(len(dataset.paths)):
        if i % 100 == 99:
            print(i, "/", len(dataset.paths))
        x, name = dataset.getImageAndLabel(i)
        x = x.cuda()

        z = net(x.unsqueeze(0))

        path = "build/" + sys.argv[1] + "/" + sys.argv[2] + "/" + name
        torch.save(z, path)
