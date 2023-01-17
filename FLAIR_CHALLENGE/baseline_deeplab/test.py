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


def largeforward(net, image, tilesize=256, stride=128):
    pred = torch.zeros(13, image.shape[1], image.shape[2]).cuda()
    for row in range(0, image.shape[1] - tilesize + 1, stride):
        for col in range(0, image.shape[2] - tilesize + 1, stride):
            tmp = net(image[:, row : row + tilesize, col : col + tilesize].unsqueeze(0))
            pred[:, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


with torch.no_grad():
    cm = torch.zeros((13, 13)).cuda()
    for i in range(len(dataset.paths)):
        x, y = dataset.getImageAndLabel(i, torchformat=True)
        x, y = x.cuda(), y.cuda()

        h, w = y.shape[0], y.shape[1]
        globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
        power2resize = torch.nn.AdaptiveAvgPool2d(((h // 128) * 128, (w // 128) * 128))
        x = power2resize(x)
        z = largeforward(net, x)
        z = globalresize(z)
        x = globalresize(x)

        _, z = z.max(0)
        cm += dataloader.confusion(y, z)

        if True:
            torchvision.utils.save_image(x / 255, "build/" + str(i) + "_x.png")
            debug = torch.stack([y, y, y], dim=0) * 255 / 13
            torchvision.utils.save_image(debug, "build/" + str(i) + "_y.png")
            debug = torch.stack([z, z, z], dim=0).float() * 255 / 13
            torchvision.utils.save_image(debug, "build/" + str(i) + "_z.png")

    print(cm)
    print(dataloader.perf(cm))

os._exit(0)
