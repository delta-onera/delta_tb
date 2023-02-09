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

        self.prepa = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/GRANDE_IMAGE/PREPAREFUSION/build/"
        self.normalize = {}
        for mode in ["RGB", "RIE", "IGE", "IEB"]:
            moyenne, variance = 0, 0
            l = os.listdir(self.prepa + mode)
            l = [name for name in l if ".tif" in name]
            for name in l:
                tmp = torch.load(self.prepa + mode + "/" + name).float()
                moyenne += float(tmp.mean())
                variance += float(torch.sqrt(tmp.var()))

            moyenne, variance = (moyenne / len(l), variance / len(l))
            self.normalize[mode] = (moyenne - 2 * variance, moyenne + 2 * variance)

    def getImageAndLabel(self, i):
        x, name = self.paths[i]
        with rasterio.open(x + ".tif") as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255) / 255.0

        x = torch.Tensor(x)
        h, w = x.shape[1], x.shape[2]
        x = [x]
        for mode in ["RGB", "RIE", "IGE", "IEB"]:
            tmp = self.prepa + mode + "/" + name+".tif"
            tmp = torch.load(tmp, map_location=torch.device("cpu")).float()
            tmp = tmp.unsqueeze(0).float()
            tmp = torch.nn.functional.interpolate(tmp, size=(h, w), mode="bilinear")
            tmp = torch.clip(
                tmp, min=self.normalize[mode][0], max=self.normalize[mode][1]
            )
            tmp = tmp - self.normalize[mode][0]
            tmp = tmp / (self.normalize[mode][1] - self.normalize[mode][0])
            x.append(tmp[0])

        x = torch.cat(x, dim=0)
        assert x.shape[0] == 57
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


class FusionNet(torch.nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.f1 = torch.nn.Conv2d(57, 57, kernel_size=17, padding=8)
        self.f2 = torch.nn.Conv2d(171, 57, kernel_size=9, padding=4)
        self.f3 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
        self.f4 = torch.nn.Conv2d(171, 256, kernel_size=1)
        self.f5 = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        z = x.clone()
        z = torch.nn.functional.leaky_relu(self.f1(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f2(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f3(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f4(z))
        return self.f5(z)
