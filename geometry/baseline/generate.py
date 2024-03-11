import torch
import PIL
from PIL import Image


class Generator:
    def __init__(self):
        self.paths = []
        self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")

    def oldCoordinate(self, r, c, a, b, d):
        dr_ = a[0] + r * d[0], a[1] + r * d[1]
        dc_ = a[0] + c * b[0], a[1] + c * b[1]
        return dr_[0] + dc_[0], dr[1] + dc_[1]

    def get(self):
        tirage = torch.rand(6)

        i = int(tirage[0] * len(self.paths))
        image = PIL.Image.open(self.paths[i])
        image = torch.Tensor(numpy.asarray(image))

        r = int(tirage[1] * image.shape[0] - 2050)
        c = int(tirage[2] * image.shape[1] - 2050)
        x = image[r : r + 2048, c : c + 2048, :].clone()
        x = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 2]], dim=0)

        a = int(tirage[3] * 1000 + 250), int(tirage[4] * 1000 + 250)

        angle = tirage[5] * 3.1415 / 2

        dbx = int(512 * torch.cos(tb))
        dby = int(512 * torch.sin(tb))
        b = a[0] + dby, a[1] + dbx

        ddx = int(512 * torch.cos(tb + 3.1415 / 2))
        ddy = int(512 * torch.sin(tb + 3.1415 / 2))
        d = a[0] + ddy, a[1] + ddx

        x_ = torch.zeros(3, 512, 512)
        for r in range(512):
            for c in range(512):
                r_, c_ = self.oldCoordinate(r, c, a, b, d)
                x_[:, r, c] = x[:, r_, c_]

        return x, x_, (a, b, d)
