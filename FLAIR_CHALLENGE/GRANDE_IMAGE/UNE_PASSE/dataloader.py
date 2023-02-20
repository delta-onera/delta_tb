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
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        return torch.Tensor(x), self.paths[i][1]

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
        self.compression = torch.nn.Conv2d(1280, 122, kernel_size=1)
        with torch.no_grad():
            tmp = torch.cat([self.f[0][0].weight.clone()] * 2, dim=1)
            self.f[0][0] = torch.nn.Conv2d(
                6, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.f[0][0].weight = torch.nn.Parameter(tmp * 0.5)

        self.f1 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.f2 = torch.nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=False)
        self.f3 = torch.nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=False)
        self.f4 = torch.nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=False)
        self.classif2 = torch.nn.Conv2d(128, 13, kernel_size=1)
        self.compression2 = torch.nn.Conv2d(128, 58, kernel_size=1)

        self.f5 = torch.nn.Conv2d(64, 58, kernel_size=3, padding=1, bias=False)
        self.f6 = torch.nn.Conv2d(64, 58, kernel_size=3, padding=1, bias=False)
        self.f7 = torch.nn.Conv2d(64, 58, kernel_size=3, padding=1, bias=False)
        self.classif3 = torch.nn.Conv2d(64, 13, kernel_size=1, bias=False)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.25

        b, ch, h, w = x.shape
        assert ch == 5
        padding = torch.ones(b, 1, h, w).cuda()
        x = torch.cat([x, padding], dim=1)
        x4 = torch.nn.functional.adaptive_avg_pool2d(x, (h // 4, w // 4))

        z = self.f(x)
        p = self.classif(z)
        p = torch.nn.functional.interpolate(p, size=(h, w), mode="bilinear")

        z = self.compression(z)
        z = torch.nn.functional.interpolate(z, size=(h // 4, w // 4), mode="bilinear")
        z0 = torch.cat([x4, z], dim=1)

        z = torch.nn.functional.leaky_relu(self.f1(z0))
        z = torch.cat([z0, z, z0 * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f2(z))
        z = torch.cat([z0, z, z0 * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f3(z))
        z = torch.cat([z0, z, z0 * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f4(z))

        p2 = self.classif2(z)
        p2 = torch.nn.functional.interpolate(p2, size=(h, w), mode="bilinear")

        z = self.compression2(z)
        z = torch.nn.functional.interpolate(z, size=(h, w), mode="bilinear")

        z = torch.cat([x, z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f5(z))
        z = torch.cat([x, z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f6(z))
        z = torch.cat([x, z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f7(z))
        z = torch.cat([x, z], dim=1)

        p3 = self.classif3(z)

        return p2 + 0.3 * p + 0.3 * p3
