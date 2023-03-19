import os
import PIL
from PIL import Image
import rasterio
import numpy
import torch
import queue
import threading
import torchvision


def confusion(y, z):
    cm = torch.zeros(13, 13).cuda()
    for a in range(13):
        for b in range(13):
            cm[a][b] = ((z == a).float() * (y == b).float()).sum()
    return cm


def perf(cm):
    cmt = torch.transpose(cm, 0, 1)

    accu = 0
    for i in range(12):
        accu += cm[i][i]
    accu /= cm[0:12, 0:12].flatten().sum()

    iou = 0
    for i in range(12):
        inter = cm[i][i]
        union = cm[i].sum() + cmt[i].sum() - cm[i][i] + (cm[i][i] == 0)
        iou += inter / union

    return (iou / 12 * 100, accu * 100)


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        y = numpy.transpose(y, axes=(1, 0))
        for u in range(x.shape[0]):
            x[u] = numpy.transpose(x[u], axes=(1, 0))
    if j == 1:
        y = numpy.flip(y, axis=0)
        for u in range(x.shape[0]):
            x[u] = numpy.flip(x[u], axis=0)
    if k == 1:
        y = numpy.flip(y, axis=1)
        for u in range(x.shape[0]):
            x[u] = numpy.flip(x[u], axis=1)
    return x.copy(), y.copy()


class CropExtractor(threading.Thread):
    def __init__(self, paths):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 161
        self.paths = paths

        self.erreur = [
            0.835,
            0.569,
            0.756,
            0.627,
            0.896,
            0.371,
            0.671,
            0.398,
            0.844,
            0.559,
            0.519,
            0.439,
        ]
        self.erreur = torch.Tensor(self.erreur)

    def getName(self, i):
        return self.paths[i][-1]

    def getImage(self, i):
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)
            x = x / 255
            x = (x - 0.5) / (0.5)

        if len(self.paths[i]) == 1:
            xx = PIL.Image.open(
                "../../cheat/orientedfusion/predictions/PRED_" + self.getName(i)
            )
            xx = numpy.asarray(xx.convert("L").copy())
            with torch.no_grad():
                xx = torch.Tensor(x).long()
                xx = torch.nn.functional.one_hot(xx, num_classes=12)
        else:
            xx = torch.Tensor(self.getLabel(i)).long()
            xx = xx * (xx < 12).float() + 11 * (xx == 12)
            zz = torch.nn.functional.one_hot(xx.long(), num_classes=12).float()
            
            for jj in range(12):
                zz[:, :, jj] *= self.erreur[jj]
            zz = zz.sum(2) * 0.8
            assert zz.shape == xx.shape

            seuil = torch.rand(xx.shape)
            randval = (torch.rand(xx.shape) * 12).long()
            xx = xx * (zz <= seuil).float() + randval * (zz > seuil).float()
            xx = torch.nn.functional.one_hot(xx.long(), num_classes=12)

        xx = xx.numpy()
        assert xx.shape==(512,512,12)
        print(xx.shape)
        xx = numpy.transpose(xx, axes=(1, 2))
        xx = numpy.transpose(xx, axes=(0, 1))

        xxx = numpy.zeros((1, 512, 512))
        x = numpy.concat([x, xx, xxx], axis=0)

        assert x.shape == (18, 512, 512)
        return x

    def getLabel(self, i):
        y = PIL.Image.open(self.paths[i][1]).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.clip(numpy.nan_to_num(y) - 1, 0, 12)

        return y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        x = torch.zeros(batchsize, 18, 512, 512)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)

        while True:
            i = int(torch.rand(1) * len(self.paths))
            image, label = self.getImage(i), self.getLabel(i)
            flag = numpy.random.randint(0, 2, size=3)
            x, y = symetrie(image.copy(), label.copy(), flag)
            x, y = torch.Tensor(x), torch.Tensor(y)
            self.q.put((x, y), block=True)


class FLAIR:
    def __init__(self, root):
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
                    if "train" in root:
                        y = prefix + "/msk/MSK_" + name
                        self.paths.append((x, y, name))
                    else:
                        self.paths.append((x, name))

        self.data = CropExtractor(self.paths)

    def getImage(self, i):
        return torch.Tensor(self.data.getImage(i))

    def getLabel(self, i):
        return torch.Tensor(self.data.getLabel(i))

    def getName(self, i):
        return self.data.getName(i)

    def getBatch(self, batchsize):
        assert self.run
        return self.data.getBatch(batchsize)

    def start(self):
        if not self.run:
            self.run = True
            self.data.start()


class UNET_EFFICIENTNET(torch.nn.Module):
    def __init__(self):
        super(UNET_EFFICIENTNET, self).__init__()
        self.f = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        with torch.no_grad():
            tmp = torch.cat([self.f[0][0].weight.clone()] * 6, dim=1)
            self.f[0][0] = torch.nn.Conv2d(
                18, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.f[0][0].weight = torch.nn.Parameter(tmp / 6)

        self.g1 = torch.nn.Conv2d(356, 256, kernel_size=5, padding=2)
        self.g2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.g3 = torch.nn.Conv2d(356, 128, kernel_size=5, padding=2)
        self.g4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.g5 = torch.nn.Conv2d(196, 128, kernel_size=5, padding=2)
        self.g6 = torch.nn.Conv2d(128, 110, kernel_size=3, padding=1)

        self.final1 = torch.nn.Conv2d(128, 110, kernel_size=1)
        self.final2 = torch.nn.Conv2d(128, 110, kernel_size=1)
        self.classif = torch.nn.Conv2d(128, 13, kernel_size=1)

    def forward(self, x):
        z2 = self.f[1](self.f[0](x))  # 32,256
        z4 = self.f[2](z2)  # 64,128
        z8 = self.f[3](z4)  # 96,64
        zf = self.f[5](self.f[4](z8))  # 224,32

        x8max = torch.nn.functional.max_pooling2D(x, kernel_size=8, stride=8, padding=0)
        x8avg = torch.nn.functional.avg_pooling2D(x, kernel_size=8, stride=8, padding=0)
        zf = torch.nn.functional.interpolate(zf, size=(64, 64), mode="bilinear")

        z = torch.cat([x8max, x8avg, zf, z8], dim=1)  # 224+18+18+96=356
        z = torch.nn.functional.leaky_relu(self.g1(z))
        z = torch.nn.functional.leaky_relu(self.g2(z))  # 256

        z = torch.nn.functional.interpolate(z, size=(128, 128), mode="bilinear")
        x8max = torch.nn.functional.max_pooling2D(x, kernel_size=4, stride=4, padding=0)
        x8avg = torch.nn.functional.avg_pooling2D(x, kernel_size=4, stride=4, padding=0)

        z = torch.cat([x8max, x8avg, z, z4], dim=1)  # 256+18+18+64=356
        z = torch.nn.functional.leaky_relu(self.g3(z))
        z = torch.nn.functional.leaky_relu(self.g4(z))  # 128

        x8max = torch.nn.functional.max_pooling2D(x, kernel_size=2, stride=2, padding=0)
        x8avg = torch.nn.functional.avg_pooling2D(x, kernel_size=2, stride=2, padding=0)
        z = torch.nn.functional.interpolate(z, size=(256, 256), mode="bilinear")

        z = torch.cat([x8max, x8avg, z, z2], dim=1)  # 128+18+18+32=196
        z = torch.nn.functional.leaky_relu(self.g5(z))
        z = torch.nn.functional.leaky_relu(self.g6(z))  # 110

        z = torch.nn.functional.interpolate(z, size=(512, 512), mode="bilinear")
        z = torch.cat([x, z], dim=1)  # 128
        z = torch.nn.functional.leaky_relu(self.final1(z))
        z = torch.cat([x, z], dim=1)  # 128
        z = torch.nn.functional.leaky_relu(self.final2(z))
        z = torch.cat([x, z], dim=1)  # 128
        z = self.classif(z)

        return z
