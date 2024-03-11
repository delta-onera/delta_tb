import torch
import numpy
import PIL
from PIL import Image


class Generator:
    def __init__(self):
        self.paths = []
        self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")
        # self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56870.tif") # too much water+container
        # self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315135_56865.tif") # almost but...
        # self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315135_56870.tif") # too much water
        self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315140_56865.tif")
        self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")

        tmp = torch.arange(65536).long()
        self.broad = ((tmp / 256).long().cuda(), (tmp % 256).long().cuda())

    def oldCoordinate(self, r, c, coord):
        ay, ax, dby, dbx, ddy, ddx = coord
        dr_ = r / 256 * ddy, r / 256 * ddx
        dc_ = c / 256 * dby, c / 256 * dbx
        return (ay + dr_[0] + dc_[0]).long(), (ax + dr_[1] + dc_[1]).long()

    def get_(self):
        tirage = torch.rand(6)

        i = int(tirage[0] * len(self.paths))
        x = PIL.Image.open(self.paths[i])
        x = torch.Tensor(numpy.asarray(x)).clone().cuda()
        x = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 2]], dim=0)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2)

        r = int(tirage[1] * (x.shape[1] - 1030))
        c = int(tirage[2] * (x.shape[2] - 1030))
        x = x[:, r : r + 1024, c : c + 1024] / 255

        a = float(tirage[3] * 400 + 300), float(tirage[4] * 400 + 300)
        angle = tirage[5] * 3.1415 / 2

        dby = float(256 * torch.sin(angle))
        dbx = float(256 * torch.cos(angle))
        ddy = float(256 * torch.sin(angle + 3.1415 / 2))
        ddx = float(256 * torch.cos(angle + 3.1415 / 2))

        coord = (a[1], a[0], dby, dbx, ddy, ddx)

        p = self.oldCoordinate(self.broad[0], self.broad[1], coord)
        p = torch.clamp(p[0], 0, 1023), torch.clamp(p[1], 0, 1023)
        x_ = torch.zeros(3, 256, 256).cuda()
        x_[0][self.broad[0], self.broad[1]] = x[0][p[0], p[1]]
        x_[1][self.broad[0], self.broad[1]] = x[1][p[0], p[1]]
        x_[2][self.broad[0], self.broad[1]] = x[2][p[0], p[1]]

        return x, x_, coord

    def get(self):
        with torch.no_grad():
            return self.get_()


if __name__ == "__main__":
    import torchvision

    gen = Generator()
    x, x_, proj = gen.get()
    print(proj)
    torchvision.utils.save_image(x, "build/source.png")
    torchvision.utils.save_image(x_, "build/target.png")

    x = PIL.Image.open(gen.paths[0])
    x = torch.Tensor(numpy.asarray(x)).clone().cuda()
    x = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 2]], dim=0)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=5)
    torchvision.utils.save_image(x / 255, "build/debug.png")
