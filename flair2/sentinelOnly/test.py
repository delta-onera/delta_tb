import torch
import dataloader
import numpy
import PIL
from PIL import Image

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

print("load data")
dataset = dataloader.FLAIR2("test")

print("test")
stats = torch.zeros((13, 13)).cuda()
with torch.no_grad():
    for name in dataset.paths:
        _, s = dataset.get(name)
        s = s.cuda()

        z = net(s.unsqueeze(0))
        _, z = z[0].max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        z = PIL.Image.fromarray(z)
        z.save("build/PRED_" + number6(name) + ".tif", compression="tiff_lzw")
