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
dataset = dataloader.FLAIR("/scratchf/flair_merged/train/", "oddodd")

print("test")


def largeforward(net, image, tilesize=256, stride=128):
    assert 512 % tilesize == 0 and tilesize % stride == 0

    pred = torch.zeros(13, image.shape[1], image.shape[2]).half().cuda()
    for row in range(0, image.shape[1] - tilesize + 1, stride):
        for col in range(0, image.shape[2] - tilesize + 1, stride):
            tmp = net(image[:, row : row + tilesize, col : col + tilesize].unsqueeze(0))
            pred[:, row : row + tilesize, col : col + tilesize] += tmp[0].half()
    return pred


with torch.no_grad():
    cm = torch.zeros((13, 13)).cuda()
    for i in range(len(dataset.paths)):
        if i % 10 == 9:
            print(i, "/", len(dataset.paths))
        x, y, _ = dataset.getImageAndLabel(i)
        x = x.cuda()

        z = largeforward(net, x)
        del x
        _, z = z.max(0).cpu()
        cm += dataloader.confusion(y, z)

    print(cm)
    print(dataloader.perf(cm))

os._exit(0)
