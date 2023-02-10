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
        self.prepa = "/d/achanhon/github/delta_tb/FLAIR_CHALLENGE/GRANDE_IMAGE/PREPAREFUSION/build/"

        wtf = 0
        maxmax = 0
        self.normalize = {}
        for mode in ["RGB", "RIE", "IGE", "IEB"]:
            moyenne, variance = 0, 0
            l = os.listdir(self.prepa + mode)
            l = [name for name in l if ".tif" in name]
            for name in l:
                tmp = torch.load(self.prepa + mode + "/" + name).float()
                tmp = torch.nn.functional.leaky_relu(tmp)
                maxmax = max(maxmax, tmp.flatten().max())

                _, lol = tmp.max(0)
                for ii in range(lol.shape[0]):
                    for jj in range(lol.shape[1]):
                        tmp[lol[ii][jj], ii, jj] = 0
                wtf = max(wtf, tmp.flatten().max())

                moyenne += float(tmp.mean())
                variance += float(torch.sqrt(tmp.var()))

            moyenne, variance = (moyenne / len(l), variance / len(l))
            self.normalize[mode] = (moyenne - 2 * variance, moyenne + 2 * variance)

        print(maxmax, wtf)

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
            tmp = torch.nn.functional.interpolate(tmp, size=(h, w), mode="bilinear")
            tmp = torch.clip(
                tmp, min=self.normalize[mode][0], max=self.normalize[mode][1]
            )
            tmp = tmp - self.normalize[mode][0]
            tmp = tmp / (self.normalize[mode][1] - self.normalize[mode][0])
            x.append(tmp[0])

        x = torch.cat(x, dim=0)
        assert x.shape == (57, y.shape[0], y.shape[1])
        if torchformat:
            return x, y, self.paths[i][2]
        else:
            return x.numpy(), y.numpy()

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


dataset = dataloader.FLAIR("/scratchf/flair_merged/train/", "oddodd")
