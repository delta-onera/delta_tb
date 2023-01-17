import os
import torch
import torchvision
import dataloader

assert torch.cuda.is_available()

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/train/", "odd")

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("test")

with torch.no_grad():
    cm = torch.zeros((13, 13)).cuda()
    for i in range(len(dataset.paths)):
        x, y = dataset.getImageAndLabel(i, torchformat=True)
        x, y = x.cuda(), y.cuda()

        z = net(x.unsqueeze(0))
        _, z = z[0].max(0)
        cm += dataloader.confusion(y, z)

        if False:
            torchvision.utils.save_image(x / 255, "build/" + str(i) + "_x.png")
            debug = torch.stack([y, y, y], dim=0) / 13
            torchvision.utils.save_image(debug, "build/" + str(i) + "_y.png")
            debug = torch.stack([z, z, z], dim=0).float() / 13
            torchvision.utils.save_image(debug, "build/" + str(i) + "_z.png")

    print(cm)
    print(dataloader.perf(cm))

os._exit(0)
