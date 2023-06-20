import torch
import dataloader
import numpy
import PIL
from PIL import Image

assert torch.cuda.is_available()

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
        x, s = dataset.get(name)
        x, s = x.cuda(), s.cuda()

        z = net(x.unsqueeze(0), s.unsqueeze(0))
        _, z = z[0].max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        z = PIL.Image.fromarray(z)
        z.save("build/" + name, compression="tiff_lzw")
