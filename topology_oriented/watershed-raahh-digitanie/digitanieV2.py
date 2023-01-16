import os
import PIL
from PIL import Image
import rasterio
import numpy
import torch
import queue
import threading
import torchvision
import skimage


def shortmaxpool(y, size=2):
    return torch.nn.functional.max_pool2d(
        y.unsqueeze(0), kernel_size=2 * size + 1, stride=1, padding=size
    )[0]


def confusion(y, z, D=None):
    if D is None:
        D = 1
    cm = torch.zeros(2, 2).cuda()
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cm[a][b] = ((z == a).float() * (y == b).float() * D).sum()
    return cm


def numpyTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def perf(cm):
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=0), numpy.flip(y, axis=0)
    return x.copy(), y.copy()


def minmax01(x):
    return numpy.clip(x * 2, 0, 1)
    xmin = numpy.min(x.flatten())
    xmax = numpy.max(x.flatten()) + 0.0000000001
    return (x - xmin) / (xmax - xmin)


def smooth(y):
    with torch.no_grad():
        yy = 1 - y
        yy = shortmaxpool(yy, size=1)  # erosion
        y = 1 - yy
        y = shortmaxpool(y, size=1)  # dilatation

        vtlabelmap = torch.Tensor(skimage.measure.label(y.numpy()))
        maxV = int(vtlabelmap.flatten().max())
        for i in range(maxV):
            tmp = (vtlabelmap == (i + 1)).float().sum()
            if tmp < 25:
                y = y * (vtlabelmap != (i + 1)).float()
        return y


class CropExtractor(threading.Thread):
    def __init__(self, paths, flag, tile=256):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 500
        self.tilesize = tile

        pathdata, pathvt, suffixvt = paths
        self.pathdata = pathdata
        self.pathvt = pathvt
        self.suffixvt = suffixvt
        assert flag in ["even", "odd", "all"]
        self.flag = flag

        if self.flag == "all":
            self.NB = 10
        else:
            self.NB = 5

    def getImageAndLabel(self, i, torchformat=False):
        assert i < self.NB

        if self.flag == "odd":
            i = i * 2 + 1
        if self.flag == "even":
            i = i * 2

        with rasterio.open(self.pathdata + str(i) + ".tif") as src:
            r = minmax01(src.read(1))
            g = minmax01(src.read(2))
            b = minmax01(src.read(3))
            x = numpy.stack([r, g, b], axis=-1)

        y = PIL.Image.open(self.pathvt + str(i) + self.suffixvt).convert("RGB").copy()
        y = numpy.asarray(y)
        y = numpy.uint8((y[:, :, 0] == 250) * (y[:, :, 1] == 50) * (y[:, :, 2] == 50))

        if torchformat:
            return numpyTOtorch(x), smooth(torch.Tensor(y))
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, self.tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize
        debug = True

        while True:
            for i in range(self.NB):
                image, label = self.getImageAndLabel(i)

                ntile = 50
                RC = numpy.random.rand(ntile, 2)
                flag = numpy.random.randint(0, 2, size=(ntile, 3))
                for j in range(ntile):
                    r = int(RC[j][0] * (image.shape[0] - tilesize - 2))
                    c = int(RC[j][1] * (image.shape[1] - tilesize - 2))
                    im = image[r : r + tilesize, c : c + tilesize, :]
                    mask = label[r : r + tilesize, c : c + tilesize]

                    if numpy.sum(numpy.int64(mask != 0)) == 0:
                        continue
                    if numpy.sum(numpy.int64(mask == 0)) == 0:
                        continue

                    x, y = symetrie(im.copy(), mask.copy(), flag[j])
                    x, y = numpyTOtorch(x), smooth(torch.Tensor(y))
                    self.q.put((x, y), block=True)


class DIGITANIE:
    def __init__(self, root, infos, flag, tile=256):
        self.run = False
        self.tilesize = tile
        self.cities = [city for city in infos]
        self.root = root
        self.NBC = len(self.cities)

        self.allimages = []
        self.data = {}
        for city in self.cities:
            pathdata = root + city + infos[city]["pathdata"]
            pathvt = root + city + "/COS9/" + city + "_"
            paths = pathdata, pathvt, infos[city]["suffixvt"]
            self.data[city] = CropExtractor(paths, flag, tile=tile)
            self.allimages.extend([(city, i) for i in range(self.data[city].NB)])
        self.NB = len(self.allimages)

    def getImageAndLabel(self, i, torchformat=False):
        city, j = self.allimages[i]
        return self.data[city].getImageAndLabel(j, torchformat=torchformat)

    def start(self):
        if not self.run:
            self.run = True
            for city in self.cities:
                self.data[city].start()

    def getBatch(self, batchsize):
        assert self.run
        batchchoice = (torch.rand(batchsize) * self.NBC).long()

        x = torch.zeros(batchsize, 3, self.tilesize, self.tilesize)
        y = torch.zeros(batchsize, self.tilesize, self.tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.data[self.cities[batchchoice[i]]].getCrop()
        return x, y


def getDIGITANIE(flag, root="/scratchf/AI4GEO/DIGITANIE/", tile=256):
    assert flag in ["odd", "even", "all"]

    infos = {}
    infos["Arcachon"] = {"pathdata": "/Arcachon_EPSG32630_", "suffixvt": "-v4.tif"}
    infos["Biarritz"] = {"pathdata": "/Biarritz_EPSG32630_", "suffixvt": "-v4.tif"}
    infos["Brisbane"] = {"pathdata": "/Brisbane_EPSG32756_", "suffixvt": "-v4.tif"}
    infos["Can-Tho"] = {"pathdata": "/Can-Tho_EPSG32648_", "suffixvt": "-v4.tif"}
    infos["Helsinki"] = {"pathdata": "/Helsinki_EPSG32635_", "suffixvt": "-v4.tif"}
    infos["Lagos"] = {"pathdata": "/Lagos_EPSG32631_", "suffixvt": "_mask.tif"}
    infos["Maros"] = {"pathdata": "/Maros_EPSG32750_", "suffixvt": "-v4.tif"}
    infos["Montpellier"] = {"pathdata": "/Montpellier_EPSG2154_", "suffixvt": "-v4.tif"}
    infos["Munich"] = {"pathdata": "/Munich_EPSG32632_", "suffixvt": "-v4.tif"}
    infos["Nantes"] = {"pathdata": "/Nantes_EPSG32630_", "suffixvt": "-v4.tif"}
    infos["Paris"] = {"pathdata": "/Paris_EPSG2154_", "suffixvt": "-v4.tif"}
    infos["Port-Elisabeth"] = {
        "pathdata": "/Port-Elisabeth_EPSG32735_",
        "suffixvt": "_mask.tif",
    }
    infos["Shanghai"] = {"pathdata": "/Shanghai_EPSG32651_", "suffixvt": "-v4.tif"}
    infos["Strasbourg"] = {"pathdata": "/Strasbourg_EPSG32632_", "suffixvt": "-v4.tif"}
    infos["Tianjin"] = {"pathdata": "/Tianjin_32650_", "suffixvt": "-v4.tif"}
    infos["Toulouse"] = {"pathdata": "/Toulouse_EPSG32631_", "suffixvt": "-v4.tif"}

    return DIGITANIE(root, infos, flag, tile=tile)


def mapfiltered(spatialmap, setofvalue):
    def myfunction(i):
        return int(int(i) in setofvalue)

    myfunctionVector = numpy.vectorize(myfunction)
    return myfunctionVector(spatialmap)


def sortmap(spatialmap):
    tmp = torch.Tensor(spatialmap)
    nb = int(tmp.flatten().max())
    tmp = sorted([(-(tmp == i).float().sum(), i) for i in range(1, nb + 1)])
    valuemap = {}
    valuemap[0] = 0
    for i, (k, j) in enumerate(tmp):
        valuemap[j] = i + 1

    def myfunction(i):
        return int(valuemap[int(i)])

    myfunctionVector = numpy.vectorize(myfunction)
    return myfunctionVector(spatialmap)


def compare(y, z):
    assert len(y.shape) == 2 and len(z.shape) == 2

    vtlabelmap, nbVT = skimage.measure.label(y, return_num=True)
    predlabelmap, nbPRED = skimage.measure.label(z, return_num=True)
    vts, preds = list(range(1, nbVT + 1)), list(range(1, nbPRED + 1))
    vtlabelmap, predlabelmap = sortmap(vtlabelmap), sortmap(predlabelmap)

    tmp1, tmp2 = vtlabelmap.flatten(), predlabelmap.flatten()
    allmatch = set(zip(list(tmp1), list(tmp2)))

    falsealarm = []
    for j in preds:
        if len([i for i in vts if (i, j) in allmatch]) != 1:
            falsealarm.append(j)
    falsealarm = set(falsealarm)
    nbFalseAlarms = len(falsealarm)
    preds = set([j for j in preds if j not in falsealarm])

    goodmatch, goodbuilding, goodpreds = [], [], []
    for i in vts:
        tmp = [j for j in preds if (i, j) in allmatch]
        if len(tmp) == 0:
            continue
        goodmatch.append((i, tmp[0]))
        goodbuilding.append(i)
        goodpreds.append(tmp[0])

    nbGOOD = len(goodpreds)
    metric = torch.Tensor([nbGOOD, nbVT, nbPRED, nbFalseAlarms])

    goodbuilding = mapfiltered(vtlabelmap, set(goodbuilding))
    goodpreds = mapfiltered(predlabelmap, set(goodpreds))
    perfect = goodbuilding * goodpreds
    vert = goodbuilding + goodpreds - perfect
    vert = vert / 2 + perfect / 2

    rouge = (1 - goodpreds) * (z != 0)
    bleu = (1 - goodbuilding) * (y != 0)

    visu = numpy.stack([rouge, vert, bleu])
    return metric, visu


def perfinstance(metric):
    nbGOOD, nbVT, nbPRED, nbFalseAlarms = metric
    recall = nbGOOD / (nbVT + 0.00001)
    precision = nbGOOD / (nbPRED + 0.00001)
    gscore = recall * precision
    return gscore, recall, precision


def computebuildingskeleton3D(y):
    yy = -shortmaxpool(-y, size=1)
    ske = [yy[i].cpu().numpy() for i in range(yy.shape[0])]
    ske = [skimage.morphology.skeletonize(m) for m in ske]
    ske = [torch.Tensor(m).cuda() for m in ske]
    return torch.stack(ske, dim=0).cuda()


class GlobalLocal(torch.nn.Module):
    def __init__(self):
        super(GlobalLocal, self).__init__()
        self.backbone = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
        self.compress = torch.nn.Conv2d(1280, 32, kernel_size=1)
        self.classiflow = torch.nn.Conv2d(1280, 2, kernel_size=1)

        self.local1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.local2 = torch.nn.Conv2d(99, 32, kernel_size=5, padding=2)
        self.local3 = torch.nn.Conv2d(99, 32, kernel_size=5, padding=2)
        self.local4 = torch.nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.local5 = torch.nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.classifhigh = torch.nn.Conv2d(32, 2, kernel_size=1)

    def forwardglobal(self, x):
        x = 2 * (x / 255) - 1
        x = torch.nn.functional.interpolate(
            x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="bilinear"
        )
        return torch.nn.functional.leaky_relu(self.backbone(x))

    def forwardlocal(self, x, f):
        z = self.local1(x)
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local2(z))
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local3(z))

        zz = torch.nn.functional.max_pool2d(z, kernel_size=3, stride=1, padding=1)
        zz = torch.nn.functional.relu(100 * z - 99 * zz)

        z = torch.cat([z, zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local4(z))
        z = torch.cat([z, z * zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local5(z))
        return self.classifhigh(z)

    def forward(self, x, mode="normal"):
        assert mode in ["normal", "globalonly", "nofinetuning"]

        if mode != "normal":
            with torch.no_grad():
                f = self.forwardglobal(x)
        else:
            f = self.forwardglobal(x)

        z = self.classiflow(f)
        z = torch.nn.functional.interpolate(
            z, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        if mode == "globalonly":
            return z

        f = torch.nn.functional.leaky_relu(self.compress(f))
        f = torch.nn.functional.interpolate(
            f, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        return self.forwardlocal(x, f) + z * 0.1
