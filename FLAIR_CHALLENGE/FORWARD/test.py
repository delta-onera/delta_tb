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


def largeforward(net, image, tilesize=256, stride=128):
    pred = torch.zeros(13, image.shape[1], image.shape[2]).cuda()
    for row in range(0, image.shape[1] - tilesize + 1, stride):
        for col in range(0, image.shape[2] - tilesize + 1, stride):
            tmp = net(image[:, row : row + tilesize, col : col + tilesize].unsqueeze(0))
            pred[:, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


with torch.no_grad():
    for i in range(len(dataset.paths)):
        if i % 100 == 99:
            print(i, "/", len(dataset.paths))
        x, name = dataset.getImageAndLabel(i)
        x = x.cuda()

        z = largeforward(z, x)
        _, z = z.max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        z = PIL.Image.fromarray(z)
        z.save("build/" + name)
