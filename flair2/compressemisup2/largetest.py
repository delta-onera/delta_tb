import torch
import dataloader
import numpy
import PIL
from PIL import Image

assert torch.cuda.is_available()
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True


def number6(i):
    s = str(i)
    while len(s) < 6:
        s = "0" + s
    return s


class DeepEnsemble(torch.nn.Module):
    def __init__(self, m1, m2):
        super(DeepEnsemble, self).__init__()
        self.m1 = torch.load(m1)
        self.m2 = torch.load(m2)

    def forward(self, x, s):
        p1 = self.m1(x, s)
        p2 = self.m2(x, s)
        return 1.1 * p1 + p2


print("load model")
net = DeepEnsemble("../semisup2/build/model.pth", "../semisup2bis/build/model.pth")
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
        z.save("build/PRED_" + number6(name) + ".tif", compression="tiff_lzw")
