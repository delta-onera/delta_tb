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
        self.maxsize = 500
        self.paths = paths

    def getImageAndLabel(self, i, torchformat=False):
        with rasterio.open(self.paths[i][0]) as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)

        y = PIL.Image.open(self.paths[i][1]).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.clip(numpy.nan_to_num(y) - 1, 0, 12)

        if torchformat:
            return torch.Tensor(x), torch.Tensor(y), self.paths[i][2]
        else:
            return x, y, self.paths[i][2]

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        x = torch.zeros(batchsize, 5, 256, 256)
        y = torch.zeros(batchsize, 256, 256)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = 256

        while True:
            i = int(torch.rand(1) * len(self.paths))
            image, label, _ = self.getImageAndLabel(i)

            ntile = 50
            RC = numpy.random.rand(ntile, 2)
            flag = numpy.random.randint(0, 2, size=(ntile, 3))
            for j in range(ntile):
                r = int(RC[j][0] * (image.shape[1] - tilesize - 2))
                c = int(RC[j][1] * (image.shape[2] - tilesize - 2))
                im = image[:, r : r + tilesize, c : c + tilesize]
                mask = label[r : r + tilesize, c : c + tilesize]
                x, y = symetrie(im.copy(), mask.copy(), flag[j])
                x, y = torch.Tensor(x), torch.Tensor(y)
                self.q.put((x, y), block=True)


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["odd", "even"]
        self.root = root
        self.flag = flag
        self.run = False
        self.domaines = os.listdir(root)
        self.paths = []
        for domaine in self.domaines:
            names = os.listdir(root + domaine)
            backup = set(names)
            names = [name[4:] for name in names if "MSK_" in name]
            names = [name for name in names if "IMG_" + name in backup]

            for name in names:
                y = root + domaine + "/MSK_" + name
                x = root + domaine + "/IMG_" + name
                self.paths.append((x, y, name))

        self.paths = sorted(self.paths)
        if flag == "even":
            tmp = [i for i in range(len(self.paths)) if i % 2 == 0]
        else:
            tmp = [i for i in range(len(self.paths)) if i % 2 == 1]
        self.paths = [self.paths[i] for i in tmp]

        self.data = CropExtractor(self.paths)

    def getImageAndLabel(self, i):
        return self.data.getImageAndLabel(i, torchformat=True)

    def getBatch(self, batchsize):
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
            tmp = torch.cat([self.f[0][0].weight.clone()] * 2, dim=1)
            self.f[0][0] = torch.nn.Conv2d(
                6, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.f[0][0].weight = torch.nn.Parameter(tmp * 0.5)




        self.classif = torch.nn.Conv2d(1280, 13, kernel_size=1)

        self.compression = torch.nn.Conv2d(1280, 238, kernel_size=1)
        self.f1 = torch.nn.Conv2d(256, 256, kernel_size=5, padding=2, bias=False)
        self.f2 = torch.nn.Conv2d(256, 256, kernel_size=5, padding=2, bias=False)
        self.f3 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.f4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.f5 = torch.nn.Conv2d(256, 256, kernel_size=1)
        self.f6 = torch.nn.Conv2d(768, 1024, kernel_size=1)
        self.f7 = torch.nn.Conv2d(1024, 13, kernel_size=3, padding=1)

    def forward(self, x):
        b, ch, h, w = x.shape
        assert ch == 5
        padding = torch.ones(b, 1, h, w).cuda()
        
        x = ((x / 255) - 0.5) / 0.5
        x = torch.cat([x, padding], dim=1)
    
        z3 = self.f[3](self.f[2](self.f[1](self.f[0](x))))  #64, 1/8, 1/8
        z4 = self.f[5](self.f[4](z3))  #160, 1/16, 1/16
        z5 = self.f[7](self.f[6](z3))  #1280, 1/32, 1/32
        
        


        z = self.compression(z)
        z = torch.nn.functional.interpolate(z, size=(h // 4, w // 4), mode="bilinear")
        p = torch.nn.functional.interpolate(p, size=(h // 4, w // 4), mode="bilinear")
        z = torch.cat([x4, xx4, xxx4, z], dim=1)

        z = z + torch.nn.functional.leaky_relu(self.f1(z))
        zz = torch.nn.functional.leaky_relu(self.f2(z))
        zz = torch.nn.functional.max_pool2d(zz, kernel_size=3, padding=1, stride=1)
        z = z + zz
        z = z + torch.nn.functional.leaky_relu(self.f3(z))
        z = z + torch.nn.functional.leaky_relu(self.f4(z))

        zz = torch.nn.functional.leaky_relu(self.f5(z))
        z = torch.cat([z, zz, zz * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f6(z))
        pp = self.f7(z)

        p = 0.1 * p + pp
        p = torch.nn.functional.interpolate(p, size=(h, w), mode="bilinear")
        return p
