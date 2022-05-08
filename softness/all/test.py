import os
import sys

if len(sys.argv) < 3:
    print("no arg provided")
    quit()
name = sys.argv[1]
size = int(sys.argv[2])

import numpy
import PIL
from PIL import Image
import torch
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()


def largeforward(net, image, tilesize=128, stride=64):
    pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).cuda()
    image = image.cuda()
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
dataset = digitanie.DigitanieALL()

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("test", name)


cm = torch.zeros((len(dataset.cities), 2, 2)).cuda()
with torch.no_grad():
    for k, city in enumerate(dataset.cities):
        print(k, city)

        for i in range(dataset.NB[city]):
            x, y = dataset.getImageAndLabel(city, i, torchformat=True)
            x, y = x.cuda(), y.cuda()

            h, w = y.shape[0], y.shape[1]
            globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
            power2resize = torch.nn.AdaptiveAvgPool2d(((h // 64) * 64, (w // 64) * 64))
            x = power2resize(x)

            z = largeforward(net, x.unsqueeze(0))
            z = globalresize(z)
            z = (z[0, 1, :, :] > z[0, 0, :, :]).float()

            cm[k] += digitanie.confusion(y, z, size=size)

            if True:
                debug = digitanie.torchTOpil(globalresize(x))
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + city + str(i) + "_x.png")
                debug = y.float()
                debug = debug * 2 * (1 - digitanie.isborder(y, size=size))
                debug = debug + digitanie.isborder(y, size=size)
                debug *= 127
                debug = debug.cpu().numpy()
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + city + str(i) + "_y.png")
                debug = z.cpu().numpy() * 255
                debug = PIL.Image.fromarray(numpy.uint8(debug))
                debug.save("build/" + city + str(i) + "_z.png")

        print("perf=", digitanie.perf(cm[k]))

perfs = digitanie.perf(cm)
print("digitanie", perfs[-1])
print(perfs)
numpy.savetxt(name, np.int16(perfs.cpu().numpy() * 1000), fmt="%i")
