import torch
import dataloader
import numpy
import PIL
from PIL import Image
import queue
import threading

assert torch.cuda.is_available()
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True


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

Q = queue.Queue(maxsize=1000)


class PushInQ(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        for name in dataset.paths:
            x, s = dataset.get(name)
            Q.put((name, x, s), block=True)


print("test")
stats = torch.zeros((13, 13)).cuda()
remplisseur = PushInQ()
remplisseur.start()

I = len(dataset.paths)
with torch.no_grad():
    while I > 0:
        if I < 32:
            name, x, s = Q.get(block=True)
            x, s = x.half().cuda(), s.half().cuda()
            z = net(x.unsqueeze(0), s.unsqueeze(0), half=True)
            _, z = z[0].max(0)

            z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
            z = PIL.Image.fromarray(z)
            z.save("build/PRED_" + number6(name) + ".tif", compression="tiff_lzw")
            I -= 1
        else:
            B = []
            for k in range(8):
                B.append(Q.get(block=True))
            X = torch.stack([x for (_, x, _) in B]).half().cuda()
            S = torch.stack([s for (_, _, s) in B]).half().cuda()
            Z = net(X, S, half=True)
            _, Z = Z.max(1)
            for k in range(8):
                z = Z[k]
                name = B[k][0]
                z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
                z = PIL.Image.fromarray(z)
                z.save("build/PRED_" + number6(name) + ".tif", compression="tiff_lzw")
            I -= 8
