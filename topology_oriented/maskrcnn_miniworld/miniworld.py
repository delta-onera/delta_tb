import os
import PIL
from PIL import Image
import numpy
import torch
import random
import queue
import threading
import torchvision
import skimage


def shortmaxpool(y, size=2):
    return torch.nn.functional.max_pool2d(
        y.unsqueeze(0), kernel_size=2 * size + 1, stride=1, padding=size
    )[0]


def confusion(y, z, D):
    cm = torch.zeros(2, 2).cuda()
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cm[a][b] = ((z == a).float() * (y == b).float() * D).sum()
    return cm


def perf(cm):
    if len(cm.shape) == 2:
        accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
        iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
        iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
        return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))
    else:
        out = torch.zeros(cm.shape[0] + 1, 4)
        for k in range(cm.shape[0]):
            out[k] = perf(cm[k])
        out[-1] = perf(torch.sum(cm, dim=0))
        return out


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=0), numpy.flip(y, axis=0)
    return x.copy(), y.copy()


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
            if tmp < 101:
                y = y * (vtlabelmap != (i + 1)).float()
        return y


class CropExtractor(threading.Thread):
    def __init__(self, path, tile=256):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 500
        self.tilesize = tile
        self.path = path

        self.NB = 0
        while os.path.exists(self.path + str(self.NB) + "_x.png"):
            self.NB += 1
        assert self.NB > 0

    def getImageAndLabel(self, i, torchformat=False):
        assert i < self.NB

        image = PIL.Image.open(self.path + str(i) + "_x.png").convert("RGB").copy()
        image = numpy.uint8(numpy.asarray(image))

        label = PIL.Image.open(self.path + str(i) + "_y.png").convert("L").copy()
        label = numpy.uint8(numpy.asarray(label))
        label = numpy.uint8(label != 0)

        if torchformat:
            return pilTOtorch(image), smooth(torch.Tensor(label))
        else:
            return image, label

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, self.tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y.long()

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize

        while True:
            I = [i for i in range(self.NB)]
            random.shuffle(I)
            for i in I:
                image, label = self.getImageAndLabel(i, torchformat=False)

                ntile = image.shape[0] * image.shape[1] // 65536 + 1
                ntile = int(min(128, ntile))

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
                    x, y = pilTOtorch(x), smooth(torch.Tensor(y))
                    self.q.put((x, y), block=True)


class MiniWorld:
    def __init__(self, infos, prefix="", suffix="", tile=256):
        self.run = False
        self.tilesize = tile
        self.cities = [city for city in infos]
        self.prefix = prefix
        self.suffix = suffix
        self.NBC = len(self.cities)

        self.allimages = []
        self.data = {}
        for city in self.cities:
            path = prefix + infos[city]["path"] + suffix
            self.data[city] = CropExtractor(path, tile=tile)
            for i in range(self.data[city].NB):
                self.allimages.append((city, i))
        self.NB = len(self.allimages)

        self.priority = numpy.ones(len(self.cities))
        for i, city in enumerate(self.cities):
            self.priority[i] += infos[city]["priority"]
        self.priority = numpy.float32(self.priority) / numpy.sum(self.priority)

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
        batchchoice = numpy.random.choice(self.NBC, batchsize, p=self.priority)

        x = torch.zeros(batchsize, 3, self.tilesize, self.tilesize)
        y = torch.zeros(batchsize, self.tilesize, self.tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.data[self.cities[batchchoice[i]]].getCrop()
        return x, y.long()  # , torch.Tensor(batchchoice).long()


def getMiniworld(flag, root="/scratchf/miniworld/", tile=256):
    assert flag in ["/train/", "/test/"]

    infos = {}
    infos["potsdam"] = {"size": "small", "label": "manual"}
    infos["bruges"] = {"size": "small", "label": "manual"}
    infos["Arlington"] = {"size": "small", "label": "osm"}
    infos["NewHaven"] = {"size": "small", "label": "osm"}
    infos["Norfolk"] = {"size": "small", "label": "osm"}
    infos["Seekonk"] = {"size": "small", "label": "osm"}
    infos["Atlanta"] = {"size": "small", "label": "osm"}
    infos["Austin"] = {"size": "small", "label": "osm"}
    infos["DC"] = {"size": "small", "label": "osm"}
    infos["NewYork"] = {"size": "small", "label": "osm"}
    infos["SanFrancisco"] = {"size": "small", "label": "osm"}
    infos["chicago"] = {"size": "medium", "label": "osm"}
    infos["kitsap"] = {"size": "medium", "label": "osm"}
    infos["austin"] = {"size": "medium", "label": "osm"}
    infos["tyrol-w"] = {"size": "medium", "label": "osm"}
    infos["vienna"] = {"size": "medium", "label": "osm"}
    infos["rio"] = {"size": "large", "label": "osm"}
    infos["christchurch"] = {"size": "large", "label": "manual"}
    infos["pologne"] = {"size": "large", "label": "manual"}
    # infos["shanghai"] = {"size": "large", "label": "osm"}
    # infos["vegas"] = {"size": "large", "label": "osm"}
    # infos["khartoum"] = {"size": "large", "label": "osm"}

    for city in infos:
        infos[city]["path"] = city
        priority = 0
        if infos[city]["label"] == "manual":
            priority += 1
        if infos[city]["size"] == "medium":
            priority += 1
        if infos[city]["size"] == "large":
            priority += 2
        infos[city]["priority"] = priority

    return MiniWorld(infos, root, flag, tile=tile)


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


def computecriticalborder2D(y, size=9):
    assert len(y.shape) == 2

    def inverseValue(y):
        ym = torch.max(y.flatten())
        return (ym + 1 - y) * (y != 0).float()

    vtlabelmap = skimage.measure.label(y)
    vtlabelmap = torch.Tensor(vtlabelmap)

    vtlabelmapE = shortmaxpool(vtlabelmap, size=size)

    Ivtlabelmap = inverseValue(vtlabelmap)
    IvtlabelmapE = shortmaxpool(Ivtlabelmap, size=size)
    vtlabelmapEbis = inverseValue(IvtlabelmapE)

    out = (vtlabelmap == 0).float() * (vtlabelmapEbis != vtlabelmapE).float()
    return torch.Tensor(out)


def computecriticalborder3D(y, size=9):
    assert len(y.shape) == 3
    with torch.no_grad():
        yy = [computecriticalborder2D(y[i], size=size) for i in range(y.shape[0])]
        return torch.stack(yy, dim=0).cuda()


def computebuildingskeleton2D(y):
    assert len(y.shape) == 2
    skeleton = skimage.morphology.skeletonize(y)

    huitV = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for k in range(2):
        row, col = skeleton.nonzero()
        rowcol = [(row[i], col[i]) for i in range(row.shape[0])]
        rowcol = set(rowcol)

        notborderskeleton = []
        for row, col in rowcol:
            voisin = [1 for dr, dc in huitV if (row + dr, col + dc) in rowcol]
            voisin = sum(voisin)
            if voisin > 1:
                notborderskeleton.append((row, col))
        skeleton = numpy.zeros(skeleton.shape)
        for row, col in notborderskeleton:
            skeleton[row][col] = 1

    skeleton = torch.Tensor(skeleton)
    skeleton = shortmaxpool(skeleton, size=1)

    y = torch.Tensor(y)
    yy = 1 - shortmaxpool(1 - y, size=1)
    skeleton = skeleton * (yy == 1).float()

    yyy = 1 - shortmaxpool(1 - y, size=7)
    return 0.1 * (yyy != 0) + skeleton


def computebuildingskeleton3D(y):
    assert len(y.shape) == 3
    with torch.no_grad():
        yy = [computebuildingskeleton2D(y[i]) for i in range(y.shape[0])]
        return torch.stack(yy, dim=0).cuda()


def getboundingbox(y):
    if (y != 0).float().sum() == 0:
        return None
    bb = y.nonzero()
    bb = bb[:, 1].min(), bb[:, 0].min(), bb[:, 1].max() + 1, bb[:, 0].max() + 1
    return torch.Tensor(list(bb))


class MaskRCNN(torch.nn.Module):
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.backend = torchvision.models.detection.maskrcnn_resnet50_fpn(
            trainable_backbone_layers=True, weights="DEFAULT"
        )

    def train(self, x, y):
        x = [x[i] / 255 for i in range(x.shape[0])]
        vt = [dict() for i in range(len(x))]

        for i in range(len(x)):
            vtlabelmap = skimage.measure.label(y[i].numpy())
            vtlabelmap = torch.Tensor(vtlabelmap)
            nbVT = int(vtlabelmap.flatten().max())
            labels, boxes = torch.ones(nbVT).long(), torch.zeros(nbVT, 4)
            masks = torch.zeros(nbVT, y.shape[1], y.shape[2])

            for j in range(nbVT):
                masks[j] = vtlabelmap == (j + 1)
                boxes[j] = getboundingbox(masks[j])
            masks = masks.type(torch.uint8)

            vt[i]["boxes"] = boxes.cuda()
            vt[i]["labels"] = labels.cuda()
            vt[i]["masks"] = masks.cuda()
        return self.backend(x, targets=vt)

    def testsingle(self, x):
        z = self.backend([x / 255])[0]
        if z["masks"].shape[0] == 0:
            tmp = torch.zeros(2, x.shape[1], x.shape[2]).cuda()
            tmp[0] = 1
            tmp[1] = -1
            return tmp

        z = z["masks"][:, 0, :, :].float()
        z = z.max(0)[0] - 0.5
        return torch.stack([-z, z], dim=0)

    def test(self, x):
        if len(x.shape) == 3:
            return self.testsingle(x)
        z = [self.testsingle(x[i]) for i in range(x.shape[0])]
        return torch.stack(z, dim=0)

    def forward(self, x=None, y=None):
        if x is None:
            return None
        if y is None:
            self.backend.eval()
            return self.test(x)
        else:
            self.backend.train()
            return self.train(x, y)
