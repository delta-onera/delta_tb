import os
import torch
import torchvision
import miniworld

assert torch.cuda.is_available()

print("load data")
dataset = miniworld.CropExtractor("/scratchf/miniworld/potsdam/test/")

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()

print("test")


def largeforward(net, image, tilesize=512, stride=256):
    pred = torch.zeros(2, image.shape[1], image.shape[2]).cuda()
    for row in range(0, image.shape[1] - tilesize + 1, stride):
        for col in range(0, image.shape[2] - tilesize + 1, stride):
            tmp = net(x=image[:, row : row + tilesize, col : col + tilesize])
            pred[:, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


with torch.no_grad():
    cm = torch.zeros((2, 2)).cuda()
    instance = torch.zeros(4)
    for i in range(dataset.NB):
        x, y = dataset.getImageAndLabel(i, torchformat=True)
        x, y, D = x.cuda(), y.cuda(), torch.ones(y.shape).cuda()

        h, w = y.shape[0], y.shape[1]
        globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
        power2resize = torch.nn.AdaptiveAvgPool2d(((h // 256) * 256, (w // 256) * 256))
        x = power2resize(x)

        print(x.shape)

        z = largeforward(net, x)
        z = globalresize(z)
        z = (z[1, :, :] > z[0, :, :]).float()

        cm += miniworld.confusion(y, z, D)
        metric, visu = miniworld.compare(y.cpu().numpy(), z.cpu().numpy())
        instance += metric

        if True:
            nextI = str(len(os.listdir("build")))
            torchvision.utils.save_image(x / 255, "build/" + nextI + "_x.png")
            debug = torch.stack([y, y, y], dim=0)
            torchvision.utils.save_image(debug, "build/" + nextI + "_y.png")
            debug = torch.stack([z, z, z], dim=0)
            torchvision.utils.save_image(debug, "build/" + nextI + "_z.png")
            debug = torch.Tensor(visu)
            torchvision.utils.save_image(debug, "build/" + nextI + "_v.png")

    print(cm)
    print(miniworld.perf(cm))
    print(instance)
    print(miniworld.perfinstance(instance))

os._exit(0)
