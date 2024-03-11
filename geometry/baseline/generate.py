import torch
import numpy
import PIL
from PIL import Image


class Generator:
    def __init__(self):
        self.paths = []
        self.paths.append("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")

        tmp = torch.arange(65536).long()
        self.broad = ((tmp / 256).long(), (tmp % 256).long())

    def oldCoordinate(self, r, c, a, b, d):
        dr_ = a[0] + r * d[0], a[1] + r * d[1]
        dc_ = a[0] + c * b[0], a[1] + c * b[1]
        return dr_[0] + dc_[0], dr_[1] + dc_[1]

    def get(self):
        tirage = torch.rand(6)

        i = int(tirage[0] * len(self.paths))
        image = PIL.Image.open(self.paths[i])
        image = image.resize((5000, 5000))
        image = torch.Tensor(numpy.asarray(image))

        r = int(tirage[1] * image.shape[0] - 1030)
        c = int(tirage[2] * image.shape[1] - 1030)
        x = image[r : r + 1024, c : c + 1024, :].clone() / 255
        x = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 2]], dim=0)

        a = int(tirage[3] * 400 + 300), int(tirage[4] * 400 + 300)

        angle = tirage[5] * 3.1415 / 2

        dbx = int(256 * torch.cos(angle))
        dby = int(256 * torch.sin(angle))
        b = a[0] + dby, a[1] + dbx

        ddx = int(256 * torch.cos(angle + 3.1415 / 2))
        ddy = int(256 * torch.sin(angle + 3.1415 / 2))
        d = a[0] + ddy, a[1] + ddx

        p = self.oldCoordinate(self.broad[0], self.broad[1], a, b, d)
        p = torch.clamp(p[0], 0, 1023), torch.clamp(p[1], 0, 1023)
        x_ = torch.zeros(3, 256, 256)
        x_[:, self.broad[0], self.broad[1]] = x[:, p[0], p[1]]

        return x, x_, (a, b, d)


if __name__ == "__main__":
    import torchvision

    gen = Generator()
    x, x_, proj = gen.get()
    print(proj)
    torchvision.utils.save_image(x, "build/source.png")
    torchvision.utils.save_image(x_, "build/target.png")
