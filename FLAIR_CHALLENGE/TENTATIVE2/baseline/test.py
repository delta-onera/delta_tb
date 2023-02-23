import os
import torch
import torchvision
import dataloader

assert torch.cuda.is_available()

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/train/", "odd")

print("test")

with torch.no_grad():
    cm = torch.zeros((13, 13)).cuda()
    for i in range(len(dataset.paths)):
        if i % 100 == 99:
            print(i, "/", len(dataset.paths))
        x, y, _ = dataset.getImageAndLabel(i)
        x, y = x.cuda(), y.cuda()

        z = net(x)
        _, z = z.max(0)
        cm += dataloader.confusion(y, z)

        if False:
            debug = torch.stack([y, y, y], dim=0) / 13
            torchvision.utils.save_image(debug, "build/" + str(i) + "_y.png")
            debug = torch.stack([z, z, z], dim=0).float() / 13
            torchvision.utils.save_image(debug, "build/" + str(i) + "_z.png")

    print(cm)
    print(dataloader.perf(cm))

os._exit(0)
