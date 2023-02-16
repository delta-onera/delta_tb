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


class CropExtractor(threading.Thread):
    def __init__(self, paths):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 500
        self.paths = paths
        self.prepa = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/GRANDE_IMAGE/PREPAREFUSION/build/"

    def getImageAndLabel(self, i, torchformat=False):
        x, y, name = self.paths[i]
        with rasterio.open(x) as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255) / 255.0

        y = PIL.Image.open(y).convert("L").copy()
        y = numpy.asarray(y)
        y = numpy.clip(numpy.nan_to_num(y) - 1, 0, 12)

        x, y = torch.Tensor(x), torch.Tensor(y)
        h, w = y.shape[0], y.shape[1]
        x = [x]
        for mode in ["RGB", "RIE", "IGE", "IEB"]:
            tmp = self.prepa + mode + "/" + name
            tmp = torch.load(tmp, map_location=torch.device("cpu")).float()
            tmp = tmp.unsqueeze(0).float()
            tmp = torch.nn.functional.leaky_relu(tmp) / 50
            tmp = torch.nn.functional.interpolate(tmp, size=(h, w), mode="bilinear")
            x.append(tmp[0])

        x = torch.cat(x, dim=0)
        assert x.shape == (57, y.shape[0], y.shape[1])
        if torchformat:
            return x, y, self.paths[i][2]
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        x = torch.zeros(batchsize, 57, 256, 256)
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
            image, label = self.getImageAndLabel(i)

            ntile = 50
            RC = torch.rand(ntile, 2)
            for j in range(ntile):
                r = int(RC[j][0] * (image.shape[1] - tilesize - 2))
                c = int(RC[j][1] * (image.shape[2] - tilesize - 2))
                im = image[:, r : r + tilesize, c : c + tilesize]
                mask = label[r : r + tilesize, c : c + tilesize]
                self.q.put((im, mask), block=True)


class FLAIR:
    def __init__(self, root, flag):
        assert flag in ["oddeven", "oddodd"]
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
                self.paths.append((x, y, domaine + "_IMG_" + name))

        self.paths = sorted(self.paths)
        tmp = [i for i in range(len(self.paths)) if i % 2 == 1]
        self.paths = [self.paths[i] for i in tmp]

        if flag == "oddeven":
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


class FusionNet(torch.nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.f1 = torch.nn.Conv2d(57, 57, kernel_size=17, padding=8)
        self.f2 = torch.nn.Conv2d(171, 57, kernel_size=9, padding=4)
        self.f3 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
        self.f31 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
        self.f32 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
        self.f33 = torch.nn.Conv2d(171, 57, kernel_size=5, padding=2)
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
        z = torch.nn.functional.leaky_relu(self.f31(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f32(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f33(z))
        z = torch.cat([x, z, x * z], dim=1)
        z = torch.nn.functional.leaky_relu(self.f4(z))
        return self.f5(z)
