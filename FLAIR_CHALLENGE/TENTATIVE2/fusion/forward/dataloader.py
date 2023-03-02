import os
import rasterio
import numpy
import torch
import torchvision


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["1/2", "2/2", "1", "2/3", "3/3"]
        self.root = root
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
        N = len(self.paths)
        if flag == "1/2":
            self.paths = [self.paths[i] for i in range(N // 2)]
        if flag == "2/2":
            self.paths = [self.paths[i] for i in range(N // 2, N)]
        if flag == "2/3":
            self.paths = [self.paths[i] for i in range(2 * N // 3)]
        if flag == "3/3":
            self.paths = [self.paths[i] for i in range(2 * N // 3, N)]

    def getImageAndLabel(self, i):
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)
        return torch.Tensor(x), self.paths[i][1]


class UNET_EFFICIENTNET(torch.nn.Module):
    def __init__(self):
        super(UNET_EFFICIENTNET, self).__init__()
        self.f = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        with torch.no_grad():
            tmp = torch.cat([self.f[0][0].weight.clone()] * 2, dim=1)
            self.f[0][0] = torch.nn.Conv2d(
                6, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.f[0][0].weight = torch.nn.Parameter(tmp * 0.5)

        self.g1 = torch.nn.Conv2d(1504, 256, kernel_size=5, padding=2)
        self.g2 = torch.nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.g3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.g4 = torch.nn.Conv2d(2112, 256, kernel_size=5, padding=2)
        self.g5 = torch.nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.g6 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(2112, 13, kernel_size=1)

    def forward(self, x):
        b, ch, h, w = x.shape
        assert ch == 5 and w == 512 and h == 512
        padding = torch.ones(b, 1, h, w).cuda()

        x = ((x / 255) - 0.5) / 0.5
        x = torch.cat([x, padding], dim=1)

        z3 = self.f[3](self.f[2](self.f[1](self.f[0](x))))
        z4 = self.f[5](self.f[4](z3))
        z5 = self.f[8](self.f[7](self.f[6](z4)))

        z5 = torch.nn.functional.interpolate(z5, size=(32, 32), mode="bilinear")
        z = torch.cat([z4, z5], dim=1)
        z = torch.nn.functional.leaky_relu(self.g1(z))
        z = torch.nn.functional.leaky_relu(self.g2(z))
        z = torch.nn.functional.leaky_relu(self.g3(z))

        z4 = torch.nn.functional.interpolate(z4, size=(64, 64), mode="bilinear")
        z5 = torch.nn.functional.interpolate(z5, size=(64, 64), mode="bilinear")
        z = torch.nn.functional.interpolate(z, size=(64, 64), mode="bilinear")
        z = torch.cat([z, z3, z4, z5], dim=1)
        z = torch.nn.functional.leaky_relu(self.g4(z))
        z = torch.nn.functional.leaky_relu(self.g5(z))
        z = torch.nn.functional.leaky_relu(self.g6(z))

        z = torch.cat([z, z3, z4, z5], dim=1)
        z = self.classif(z)

        z = torch.nn.functional.interpolate(z, size=(h, w), mode="bilinear")
        return z
