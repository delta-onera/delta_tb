import torch
import PIL
from PIL import Image


class Generator:
    def __init__(self):
        self.paths = []
        self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")

        tmp = torch.arange(65536).long()
        self.broad = torch.zeros(65536, 2).long()
        self.broad[0] = (tmp / 256).long()
        self.broad[1] = (tmp % 256).long()

    def oldCoordinate(self, r, c, a, b, d):
        dr_ = a[0] + r * d[0], a[1] + r * d[1]
        dc_ = a[0] + c * b[0], a[1] + c * b[1]
        return dr_[0] + dc_[0], dr[1] + dc_[1]

    def get(self):
        tirage = torch.rand(6)

        i = int(tirage[0] * len(self.paths))
        image = PIL.Image.open(self.paths[i])
        image = image.resize((5000, 5000))
        image = torch.Tensor(numpy.asarray(image))

        r = int(tirage[1] * image.shape[0] - 1030)
        c = int(tirage[2] * image.shape[1] - 1030)
        x = image[r : r + 1024, c : c + 1024, :].clone()
        x = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 2]], dim=0)

        a = int(tirage[3] * 400 + 300), int(tirage[4] * 400 + 300)

        angle = tirage[5] * 3.1415 / 2

        dbx = int(256 * torch.cos(tb))
        dby = int(256 * torch.sin(tb))
        b = a[0] + dby, a[1] + dbx

        ddx = int(256 * torch.cos(tb + 3.1415 / 2))
        ddy = int(256 * torch.sin(tb + 3.1415 / 2))
        d = a[0] + ddy, a[1] + ddx

        p = self.oldCoordinate(self.broad[0], self.broad[1], a, b, d)
        x_ = torch.zeros(3, 512, 512)
        x_[:, self.broad[0], self.broad[1]] = x[:, p[0], p[1]]

        return x, x_, (a, b, d)
