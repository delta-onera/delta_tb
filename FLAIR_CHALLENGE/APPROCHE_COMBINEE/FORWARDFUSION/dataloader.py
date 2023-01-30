import os
import PIL
from PIL import Image
import rasterio
import numpy
import torch
import queue
import threading
import torchvision


class FLAIRTEST:
    def __init__(self, root):
        self.root = root

        self.paths = []
        level1 = os.listdir(root)
        for folder in level1:
            level2 = os.listdir(root + folder)

            for subfolder in level2:
                path = root + folder + "/" + subfolder
                level3 = os.listdir(path + "/img")
                level3 = [name[4:] for name in level3 if ".aux" not in name]
                level3 = [name[0:-4] for name in level3]

                for name in level3:
                    x = path + "/img/IMG_" + name + ".tif"
                    name = "PRED_" + name + ".tif"
                    meta = None
                    self.paths.append((x, name, meta))

    def getImageAndLabel(self, i):
        x, name, _ = self.paths[i]
        with rasterio.open(x) as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)
            x = torch.Tensor(x)
            h, w = 512, 512
            x = [x]
            for mode in ["RGB", "RIE", "IGE", "IEB"]:
                path = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/APPROCHE_COMBINEE/PREPAREFUSION/build/"
                tmp = path + mode + "/test/" + name
                tmp = torch.load(tmp, map_location=torch.device("cpu"))
                tmp = tmp.unsqueeze(0).float()
                tmp = torch.nn.functional.interpolate(tmp, size=(h, w), mode="bilinear")
                x.append(tmp[0])

        x = torch.cat(x, dim=0)
        assert x.shape == (57, 512, 512)
        return x, name


class FusionNet(torch.nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.f1 = torch.nn.Conv2d(57, 57, kernel_size=17, padding=8)
        self.f2 = torch.nn.Conv2d(171, 57, kernel_size=9, padding=4)
        self.f3 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
        self.f4 = torch.nn.Conv2d(171, 256, kernel_size=1)
        self.f5 = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        z = x / 255
        z = torch.nn.functional.leaky_relu(self.f1(z))
        z = torch.cat([x, z, x * z * 0.1], dim=1)
        z = torch.nn.functional.leaky_relu(self.f2(z))
        z = torch.cat([x, z, x * z * 0.1], dim=1)
        z = torch.nn.functional.leaky_relu(self.f3(z))
        x = torch.cat([x, z, x * z * 0.1], dim=1)
        x = torch.nn.functional.leaky_relu(self.f4(x))
        return self.f5(x)
