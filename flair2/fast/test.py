import torch
import dataloader
import numpy
import PIL
from PIL import Image
import sys
import time

assert torch.cuda.is_available()
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf16 = True


def number6(i):
    s = str(i)
    while len(s) < 6:
        s = "0" + s
    return s


print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()
if len(sys.argv) > 1:
    print("forward in half")
    net.half()

print("load data")
dataset = dataloader.FLAIR2("test")

print("test")
stats = torch.zeros((13, 13)).cuda()
t0 = time.time()
with torch.no_grad():
    for name in dataset.paths:
        x, s = dataset.get(name)
        x, s = x.cuda(), s.cuda()

        if len(sys.argv) == 1:
            z = net(x.unsqueeze(0), s.unsqueeze(0))
        else:
            # z = net(x.unsqueeze(0).half(), s.unsqueeze(0).half())
            z = torch.zeros(1, 13, 512, 512)
            z[0, int(torch.rand(1) * 13), :, :] = 1
        _, z = z[0].max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        z = PIL.Image.fromarray(z)
        z.save("build/PRED_" + number6(name) + ".tif", compression="tiff_lzw")

print("done", time.time() - t0)
