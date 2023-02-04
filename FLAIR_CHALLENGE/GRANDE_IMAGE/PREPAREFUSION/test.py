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

root = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/GRANDE_IMAGE/"
with torch.no_grad():
    net = torch.load(root + sys.argv[1] + "/build/model.pth")
    net = net.cuda()
    net.eval()

print("load data")
dataset = dataloader.ALLFLAIR(
    "/scratchf/flair_merged/" + sys.argv[2] + "/", net.channels
)


print("test")


def largeforward(net, image, tilesize=256, stride=128):
    assert 512 % tilesize == 0 and tilesize % stride == 0

    pred = torch.zeros(13, image.shape[1], image.shape[2]).cuda()
    for row in range(0, image.shape[1] - tilesize + 1, stride):
        for col in range(0, image.shape[2] - tilesize + 1, stride):
            tmp = net(image[:, row : row + tilesize, col : col + tilesize].unsqueeze(0))
            pred[:, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


with torch.no_grad():
    for i in range(len(dataset.paths)):
        print(i, "/", len(dataset.paths))
        x, name = dataset.getImageAndLabel(i)
        x = x.cuda()

        z = largeforward(net, x)
        z = z[0].half()

        torch.save(z, "build/" + sys.argv[1] + "/" + name)
