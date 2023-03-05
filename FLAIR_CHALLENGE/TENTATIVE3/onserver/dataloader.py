import os
import rasterio
import numpy
import torch
import torchvision


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["1"]
        self.root = root
        self.run = False
        self.paths = []

        domaines = os.listdir(root)
        for domaine in domaines:
            sousdomaines = os.listdir(root + domaine)
            for sousdomaines in sousdomaines:
                prefix = root + domaine + "/" + sousdomaines
                names = os.listdir(prefix + "/img")
                names = [name for name in names if ".aux" not in name]
                names = [name[4:] for name in names if "IMG_" in name]

                for name in names:
                    x = prefix + "/img/IMG_" + name
                    self.paths.append((x, name))

        self.paths = sorted(self.paths)

    def getImageAndLabel(self, i):
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)
            x = torch.Tensor(x) / 255

        x = [x]
        for mode in ["model1/", "model2/", "model3/", "model4/"]:
            path = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/TENTATIVE2/fusion/generatetest/build/"
            tmp = path + mode + self.getName(i) + ".pth"
            tmp = torch.load(tmp, map_location=torch.device("cpu"))
            tmp = tmp.unsqueeze(0).float()
            tmp = torch.nn.functional.interpolate(tmp, size=(512, 512), mode="bilinear")
            x.append(tmp[0] / 15)

        x = torch.cat(x, dim=0)
        assert x.shape == (57, 512, 512)
        return x

    def getName(self, i):
        return self.paths[i][1]


class FUSION(torch.nn.Module):
    def __init__(self):
        super(FUSION, self).__init__()
        self.f1 = torch.nn.Conv2d(57, 57, kernel_size=17, padding=8)
        self.f2 = torch.nn.Conv2d(171, 57, kernel_size=9, padding=4)
        self.f3 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
        self.f4 = torch.nn.Conv2d(171, 256, kernel_size=1)
        self.f5 = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        z = torch.nn.functional.leaky_relu(self.f1(x))
        z = torch.cat([x, z, x * z * 0.1], dim=1)
        z = torch.nn.functional.leaky_relu(self.f2(z))
        z = torch.cat([x, z, x * z * 0.1], dim=1)
        z = torch.nn.functional.leaky_relu(self.f3(z))
        x = torch.cat([x, z, x * z * 0.1], dim=1)
        x = torch.nn.functional.leaky_relu(self.f4(x))
        return self.f5(x)
