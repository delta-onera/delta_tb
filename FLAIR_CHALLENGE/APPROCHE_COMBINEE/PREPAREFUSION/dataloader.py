import os
import rasterio
import numpy
import torch
import torchvision


class FLAIRTEST:
    def __init__(self, root, channels):
        self.root = root
        self.channels = channels

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
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            x = x[self.channels]
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        # self.path[i][2] contient metadata à ajouter à x ?

        return torch.Tensor(x), self.paths[i][1]


class Mobilenet(torch.nn.Module):
    def __init__(self):
        super(Mobilenet, self).__init__()
        self.backend = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights="DEFAULT"
        )
        self.backend.classifier.low_classifier = torch.nn.Conv2d(40, 13, kernel_size=1)
        self.backend.classifier.high_classifier = torch.nn.Conv2d(
            128, 13, kernel_size=1
        )
        self.channels = None

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25
        return self.backend(x)["out"]


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
