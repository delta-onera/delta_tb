import os
import torch
import torchvision
import dataloader
import PIL
from PIL import Image

assert torch.cuda.is_available()

print("load data")
dataset = dataloader.FLAIRTEST("/scratchf/CHALLENGE_IGN/test/")

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("test")

with torch.no_grad():
    for i in range(len(dataset.paths)):
        print(i, "/", len(dataset.paths))
        x, name = dataset.getImageAndLabel(i)
        x = x.cuda()

        z = net(x.unsqueeze(0))
        _, z = z[0].max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        z = PIL.Image.fromarray(z)
        z.save("build/" + name)
