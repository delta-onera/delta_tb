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
            names = os.listdir(root + domaine)
            backup = set(names)
            names = [name[0:-4] for name in names if ".tif" in name]
            names = [name for name in names if (name + ".npy") in backup]

            for name in names:
                self.paths.append((root + domaine + "/" + name, name))

    def getImageAndLabel(self, i):
        with rasterio.open(self.paths[i][0] + ".tif") as src_img:
            x = src_img.read()
            x = x[self.channels]
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        return torch.Tensor(x), self.paths[i][1]

    def exportresults(self, i, pred):
        proj = rasterio.open(self.paths[i][0] + ".tif")
        boxes = numpy.load(self.paths[i][0] + ".npy", allow_pickle=True)

        for j in range(boxes.shape[0]):
            name = boxes[j][0]
            left, bottom, right, top = (
                boxes[j][1].left,
                boxes[j][1].bottom,
                boxes[j][1].right,
                boxes[j][1].top,
            )
            window = proj.window(left, bottom, right, top)
            col, row, w, h = window.col_off, window.row_off, window.width, window.height
            col, row, w, h = int(col), int(row), int(w), int(h)

            assert w == 512 and h == 512

            out = pred[row : row + h, left : left + w]
            out = numpy.uint8(numpy.clip(out, 0, 12))
            out = PIL.Image.fromarray(out)
            out.save("build/PRED_0" + str(name) + ".tif", compression="tiff_lzw")


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
