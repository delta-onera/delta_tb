import os
import rasterio
import torch
import torchvision
import numpy
import PIL


class FLAIRTEST:
    def __init__(self, root, channels):
        self.root = root
        self.channels = channels

        self.domaines = os.listdir(root)
        self.paths = []
        for domaine in self.domaines:
            tmp = os.listdir(root + domaine)
            backup = set(tmp)
            tmp = [name[0:-4] for name in tmp if ".tif" in name]
            tmp = [name for name in tmp if (name + ".npy") in backup]

            self.paths.append((root + domaine + "/" + name, name))

    def getImageAndLabel(self, i):
        with rasterio.open(self.paths[i][0] + ".tif") as src_img:
            x = src_img.read()
            x = x[self.channels]
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        return torch.Tensor(x), name

    def rankfromlist(self, l):
        l = list(set(l))
        out = {}
        for i in range(len(l)):
            out[l[i]] = i * 512
        return out

    def exportresults(self, i, pred):
        boxes = numpy.load(self.paths[i][0] + ".npy", allow_pickle=True)

        cols = [boxes[i][1].left for i in range(boxes.shape[0])]
        rows = [boxes[i][1].top for i in range(boxes.shape[0])]
        cols, rows = rankfromlist(cols), rankfromlist(rows)

        for j in range(boxes.shape[0]):
            name, top, left = boxes[j][0], boxes[j][1].top, boxes[j][1].left
            top, left = rows[top], cols[left]

            out = numpy.uint8(numpy.clip(pred.cpu().numpy(), 0, 12))
            out = PIL.Image.fromarray(out)
            out.save("build/PRED_" + name + ".tif", compression="tiff_lzw")


class JustEfficientnet(torch.nn.Module):
    def __init__(self):
        super(JustEfficientnet, self).__init__()
        self.f = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
        self.classif = torch.nn.Conv2d(1280, 13, kernel_size=1)
        self.channels = None

    def forward(self, x):
        _, _, h, w = x.shape
        x = ((x / 255) - 0.5) / 0.25
        x = self.f(x)
        x = self.classif(x)
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear")
        return x
