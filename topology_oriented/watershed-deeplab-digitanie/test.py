import os
import torch
import torchvision
import digitanieV2

assert torch.cuda.is_available()

print("load data")
dataset = digitanieV2.getDIGITANIE("odd")

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("test")


def largeforward(net, image, tilesize=256, stride=128):
    pred = torch.zeros(2, image.shape[1], image.shape[2]).cuda()
    for row in range(0, image.shape[1] - tilesize + 1, stride):
        for col in range(0, image.shape[2] - tilesize + 1, stride):
            tmp = net(image[:, row : row + tilesize, col : col + tilesize].unsqueeze(0))
            pred[:, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


with torch.no_grad():
    cm = torch.zeros((2, 2)).cuda()
    instance = torch.zeros(4)
    for i in range(dataset.NB):
        x, y = dataset.getImageAndLabel(i, torchformat=True)
        x, y = x.cuda(), y.cuda()

        h, w = y.shape[0], y.shape[1]
        globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
        power2resize = torch.nn.AdaptiveAvgPool2d(((h // 128) * 128, (w // 128) * 128))
        x = power2resize(x)
        z = largeforward(net, x)
        z = globalresize(z)
        x = globalresize(x)

        z = (z[1, :, :] > z[0, :, :]).float()

        cm += digitanieV2.confusion(y, z)
        metric, visu = digitanieV2.compare(y.cpu().numpy(), z.cpu().numpy())
        instance += metric

        if True:
            nextI = str(len(os.listdir("build")))
            torchvision.utils.save_image(x, "build/" + nextI + "_x.png")
            debug = torch.stack([y, y, y], dim=0)
            torchvision.utils.save_image(debug, "build/" + nextI + "_y.png")
            debug = torch.stack([z, z, z], dim=0)
            torchvision.utils.save_image(debug, "build/" + nextI + "_z.png")
            debug = torch.Tensor(visu)
            torchvision.utils.save_image(debug, "build/" + nextI + "_v.png")

    print(cm)
    print(digitanieV2.perf(cm))
    print(instance)
    print(digitanieV2.perfinstance(instance))

os._exit(0)
