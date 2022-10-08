import os
import torch
import torchvision

assert torch.cuda.is_available()


def largeforward(net, image, tilesize=128, stride=64):
    pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).cuda()
    for row in range(0, image.shape[2] - tilesize + 1, stride):
        for col in range(0, image.shape[3] - tilesize + 1, stride):
            tmp = net(image[:, :, row : row + tilesize, col : col + tilesize])
            pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
sys.path.append("/d/achanhon/github/pytorch-image-models")
sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
sys.path.append("/d/achanhon/github/segmentation_models.pytorch")

import segmentation_models_pytorch as smp
import miniworld

print("load data")
dataset = miniworld.CropExtractor("/scratchf/miniworld/potsdam/test/")

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("test")
with torch.no_grad():
    cm = torch.zeros((2, 2)).cuda()
    for i in range(dataset.NB):
        x, y = dataset.getImageAndLabel(i, torchformat=True)
        x, y, D = x.cuda(), y.cuda(), torch.ones(y.shape).cuda()

        h, w = y.shape[0], y.shape[1]
        globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
        power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
        x = power2resize(x)

        z = largeforward(net, x.unsqueeze(0))
        z = globalresize(z)
        z = (z[0, 1, :, :] > z[0, 0, :, :]).float()

        cm += miniworld.confusion(y, z, D)

        if True:
            nextI = str(len(os.listdir("build")))
            torchvision.save_images("build/" + nextI + "_x.png", x / 255)
            debug = torch.stack([y, y, y], dim=0)
            torchvision.save_images("build/" + nextI + "_y.png", debug)
            debug = torch.stack([z, z, z], dim=0)
            torchvision.save_images("build/" + nextI + "_z.png", debug)

    print(cm)
    print(dataset.perf(cm))

os._exit(0)
