import os
import rasterio
import numpy
import torch
import torchvision


class FLAIRTEST:
    def __init__(self, root, flag=""):
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
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            if onlycolor in flag:
                x = x[0:3, :, :]  # pour le moment
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

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25
        return self.backend(x)["out"]


class Mobilenet5(torch.nn.Module):
    def __init__(self, path):
        super(Mobilenet5, self).__init__()
        self.backend = torch.load(path).backend

        with torch.no_grad():
            old = self.backend.backbone["0"][0].weight.data.clone()
            self.backend.backbone["0"][0] = torch.nn.Conv2d(
                5, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            neww = self.backend.backbone["0"][0].weight.data.clone()

            neww[:, 0:3, :, :] = old
            neww[:, 3:, :, :] *= 0.5
            self.backend.backbone["0"][0].weight = torch.nn.Parameter(neww)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25
        return self.backend(x)["out"]


class Deeplab(torch.nn.Module):
    def __init__(self):
        super(Deeplab, self).__init__()
        self.backend = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DEFAULT"
        )
        self.backend.classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25
        return self.backend(x)["out"]


class GlobalLocal(torch.nn.Module):
    def __init__(self):
        super(GlobalLocal, self).__init__()
        self.backbone = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        self.compress = torch.nn.Conv2d(1280, 32, kernel_size=1)
        self.classiflow = torch.nn.Conv2d(1280, 13, kernel_size=1)

        self.local1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.local2 = torch.nn.Conv2d(99, 32, kernel_size=5, padding=2)
        self.local3 = torch.nn.Conv2d(99, 32, kernel_size=5, padding=2)
        self.local4 = torch.nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.local5 = torch.nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.classifhigh = torch.nn.Conv2d(32, 13, kernel_size=1)

    def forwardglobal(self, x):
        x = 2 * (x / 255) - 1
        x = torch.nn.functional.interpolate(
            x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="bilinear"
        )
        return torch.nn.functional.leaky_relu(self.backbone(x))

    def forwardlocal(self, x, f):
        z = self.local1(x)
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local2(z))
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local3(z))

        zz = torch.nn.functional.max_pool2d(z, kernel_size=3, stride=1, padding=1)
        zz = torch.nn.functional.relu(100 * z - 99 * zz)

        z = torch.cat([z, zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local4(z))
        z = torch.cat([z, z * zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local5(z))
        return self.classifhigh(z)

    def forward(self, x, mode="normal"):
        assert mode in ["normal", "globalonly", "nofinetuning"]

        if mode != "normal":
            with torch.no_grad():
                f = self.forwardglobal(x)
        else:
            f = self.forwardglobal(x)

        z = self.classiflow(f)
        z = torch.nn.functional.interpolate(
            z, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        if mode == "globalonly":
            return z

        f = torch.nn.functional.leaky_relu(self.compress(f))
        f = torch.nn.functional.interpolate(
            f, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        return self.forwardlocal(x, f) + z
