import os
import rasterio
import torch
import torchvision
import numpy
import PIL


class FLAIRTEST:
    def __init__(self, root):
        self.root = root

        self.domaines = os.listdir(root)
        self.paths = []
        for domaine in self.domaines:
            names = os.listdir(root + domaine)
            backup = set(names)
            names = [name[0:-4] for name in names if ".tif" in name]
            names = [name for name in names if (name + ".npy") in backup]

            for name in names:
                self.paths.append((root + domaine + "/" + name, domaine + "_" + name))

    def getImageAndLabel(self, i):
        x, name = self.paths[i]
        with rasterio.open(x + ".tif") as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255) / 255.0

        return x, self.paths[i][1]

    def exportresults(self, i, pred):
        proj = rasterio.open(self.paths[i][0] + ".tif")
        boxes = numpy.load(self.paths[i][0] + ".npy", allow_pickle=True)

        for j in range(boxes.shape[0]):
            name = boxes[j][0]
            bb = boxes[j][1]
            left, bottom, right, top = bb.left, bb.bottom, bb.right, bb.top
            window = proj.window(left, bottom, right, top)
            col, row, w, h = window.col_off, window.row_off, window.width, window.height
            col, row, w, h = round(col), round(row), round(w), round(h)

            if w != 512 or h != 512:
                print(w, h, boxes[j], window, self.paths[i][0])

            out = pred[row : row + 512, col : col + 512]
            out = numpy.uint8(numpy.clip(out, 0, 12))
            out = PIL.Image.fromarray(out)
            out.save("build/PRED_0" + str(name) + ".tif", compression="tiff_lzw")


class JustEfficientnet(torch.nn.Module):
    def __init__(self):
        super(JustEfficientnet, self).__init__()
        self.f = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        self.classif = torch.nn.Conv2d(1280, 13, kernel_size=1)
        self.compression = torch.nn.Conv2d(1280, 26, kernel_size=1)

        with torch.no_grad():
            tmp = torch.cat([self.f[0][0].weight.clone()] * 2, dim=1)
            self.f[0][0] = torch.nn.Conv2d(
                6, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.f[0][0].weight = torch.nn.Parameter(tmp * 0.5)

        self.f1 = torch.nn.Conv2d(32, 32, kernel_size=17, padding=8, bias=False)
        self.f2 = torch.nn.Conv2d(96, 32, kernel_size=9, padding=4, bias=False)
        self.f3 = torch.nn.Conv2d(96, 32, kernel_size=5, padding=2, bias=False)
        self.f31 = torch.nn.Conv2d(96, 32, kernel_size=5, padding=2, bias=False)
        self.f32 = torch.nn.Conv2d(96, 32, kernel_size=5, padding=2, bias=False)
        self.f33 = torch.nn.Conv2d(96, 32, kernel_size=5, padding=2, bias=False)
        self.f4 = torch.nn.Conv2d(96, 256, kernel_size=1, bias=False)
        self.f5 = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25

        b, ch, h, w = x.shape
        assert ch == 5
        padding = torch.ones(b, 1, h, w).cuda()
        x = torch.cat([x, padding], dim=1)

        z = self.f(x)
        p = self.classif(z)
        p = torch.nn.functional.interpolate(p, size=(h, w), mode="bilinear")
        z = self.compression(z)
        z = torch.nn.functional.interpolate(z, size=(h, w), mode="bilinear")

        x = torch.cat([x, z], dim=1)

        z = torch.nn.functional.leaky_relu(self.f1(x))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f2(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f3(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f31(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f32(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f33(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f4(z))

        return self.f5(z) + 0.1 * p
