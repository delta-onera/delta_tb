import os
import rasterio
import torch
import torchvision
import numpy


class ALLFLAIR:
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
                self.paths.append((root + domaine + "/" + name, domaine + "_" + name))

        self.paths = sorted(self.paths)
        if "/train/" in root:
            tmp = [i for i in range(len(self.paths)) if i % 2 == 1]
        self.paths = [self.paths[i] for i in tmp]

    def getImageAndLabel(self, i):
        with rasterio.open(self.paths[i][0] + ".tif") as src_img:
            x = src_img.read()
            x = x[self.channels]
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        return torch.Tensor(x), self.paths[i][1]


class JustEfficientnet(torch.nn.Module):
    def __init__(self):
        super(JustEfficientnet, self).__init__()
        self.f = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        self.classif = torch.nn.Conv2d(1280, 13, kernel_size=1)
        self.channels = None

    def forward(self, x):
        _, _, h, w = x.shape
        x = ((x / 255) - 0.5) / 0.25
        x = self.f(x)
        x = self.classif(x)
        # x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear")
        return x
