import torch
import dataloader
import numpy
import PIL
from PIL import Image
import sys

assert torch.cuda.is_available()


def number6(i):
    s = str(i)
    while len(s) < 6:
        s = "0" + s
    return s


print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()

print("get fusion rules")
if len(sys.argv)>13:
    net.w = torch.Tensor(sys.argv[1:]).cuda)()

print("load data")
dataset = dataloader.FLAIR2("test")

print("test")
stats = torch.zeros((13, 13)).cuda()
with torch.no_grad():
    for name in dataset.paths:
        x, s = dataset.get(name)
        x, s = x.cuda(), s.cuda()

        P = net(x.unsqueeze(0), s.unsqueeze(0))
        z = net.merge(P)
        _, z = z[0].max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        z = PIL.Image.fromarray(z)
        z.save("build/PRED_" + number6(name) + ".tif", compression="tiff_lzw")
