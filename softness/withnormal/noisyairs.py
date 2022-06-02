import os
import PIL
from PIL import Image, ImageDraw
import numpy
import torch
import random
import queue
import threading
from skimage.measure import find_contours, approximate_polygon


def maxpool(y, size):
    if size == 0:
        return y

    if len(y.shape) == 2:
        yy = y.unsqueeze(0).float()
    else:
        yy = y.float()

    ks = 2 * size + 1
    yyy = torch.nn.functional.max_pool2d(yy, kernel_size=ks, stride=1, padding=size)

    if len(y.shape) == 2:
        return yyy[0]
    else:
        return yyy


def isborder(y, size):
    y0, y1 = (y == 0).float(), (y == 1).float()
    y00, y11 = maxpool(y0, size=size), maxpool(y1, size=size)
    border = (y1 * y00 + y0 * y11) > 0
    return border.float()


def confusion(y, z, size):
    D = 1 - isborder(y, size=size)
    cm = torch.zeros(2, 2).cuda()
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cm[a][b] = torch.sum((z == a).float() * (y == b).float() * D)
    return cm


def perf(cm):
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def torchTOpil(x):
    return numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=0), numpy.flip(y, axis=0)
    return x.copy(), y.copy()


def replacebybbox(b, size=11):
    xmin = min(b[:, 0])
    xmax = max(b[:, 0])
    ymin = min(b[:, 1])
    ymax = max(b[:, 1])
    if xmax - xmin <= size or ymax - ymin <= size:
        out = numpy.zeros((5, 2))
        out[0] = numpy.asarray([xmin, ymin])
        out[1] = numpy.asarray([xmin, ymax])
        out[2] = numpy.asarray([xmax, ymax])
        out[3] = numpy.asarray([xmax, ymin])
        out[4] = numpy.asarray([xmin, ymin])
        return out
    else:
        return b


########################################################################
######################## CLASSIC/PUBLIC DATASETS #######################


class CropExtractor(threading.Thread):
    def __init__(self, path, maxsize=500, tilesize=128):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = maxsize

        self.path = path
        self.NB = 0
        self.tilesize = tilesize
        while os.path.exists(self.path + str(self.NB) + "_x.png"):
            self.NB += 1

        if self.NB == 0:
            print("wrong path", self.path)
            quit()

    def getImageAndLabel(self, i, torchformat=False):
        assert i < self.NB

        image = PIL.Image.open(self.path + str(i) + "_x.png").convert("RGB").copy()
        image = numpy.uint8(numpy.asarray(image))

        label = PIL.Image.open(self.path + str(i) + "_y.png").convert("L").copy()
        label = numpy.uint8(numpy.asarray(label))
        label = numpy.uint8(label != 0)

        if torchformat:
            return pilTOtorch(image), torch.Tensor(label)
        else:
            return image, label

    ###############################################################

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, self.tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        tangent = torch.zeros(batchsize, 2, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i], tangent[i] = self.getCrop()
        return x, y.long(), tangent

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
                    x, y = symetrie(im.copy(), mask.copy(), flag[j])

                    building = find_contours(y, level=0.5)
                    compress = [approximate_polygon(b, tolerance=2.5) for b in building]
                    compress = [replacebybbox(b) for b in compress]

                    tangentmask = Image.new("RGB", (tilesize, tilesize))
                    draw = ImageDraw.Draw(tangentmask)

                    for bord in compress:
                        for k in range(bord.shape[0] - 1):
                            line = list(bord[k + 1]) + list(bord[k])
                            line = line[::-1]

                            dy = line[3] - line[1]
                            dx = line[2] - line[0]
                            norm = math.sqrt(dy * dy + dx * dx + 0.00001)
                            dx, dy = dx / norm, dy / norm
                            dx, dy = int((dx + 1) * 127), int((dy + 1) * 127)

                            angleInDegrees = (254, dx, dy)
                            draw.line(xy=line, fill=angleInDegrees, width=2)

                    tang = numpy.uint8(numpy.asarray(tangentmask))

                    x, y, tang = pilTOtorch(x), torch.Tensor(y), pilTOtorch(tang)
                    self.q.put((x, y, tang), block=True)


class AIRS:
    def __init__(self, flag, tilesize=128, custom=None):
        assert flag in ["/train/", "/test/"]

        self.tilesize = tilesize
        self.root = "build/christchurch"

        assert "train" in os.listdir(self.root)
        assert "test" in os.listdir(self.root)

        self.data = CropExtractor(self.root + flag, tilesize=tilesize)
        self.run = False

    def start(self):
        if not self.run:
            self.run = True
            self.data.start()

    def getBatch(self, batchsize):
        assert self.run

        x = torch.zeros(batchsize, 3, self.tilesize, self.tilesize)
        y = torch.zeros(batchsize, self.tilesize, self.tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.data.getCrop()
        return x, y.long()


import random
import sys


def hallucination(label):
    RC = numpy.random.rand(8, 2)
    RC[:, 0] *= label.shape[0] - 9
    RC[:, 1] *= label.shape[1] - 9
    for j in range(RC.shape[0]):
        r, c = int(RC[j][0]), int(RC[j][1])
        label[r : r + 8, c : c + 8] = 1 - label[r : r + 8, c : c + 8]
    return label


def pm1image(label):
    if random.randint(0, 1) == 0:
        label = maxpool(label, size=1)
    else:
        label = 1 - label
        label = maxpool(label, size=1)
        label = 1 - label
    return label


def pm1translation(label):
    dxdy = [(-1, 1), (-1, -1), (1, 1), (1, -1)]
    tmp = random.randint(0, 100) % 4
    dx, dy = dxdy[tmp]

    h, w = label.shape
    labelbis = torch.zeros((h + 3, w + 3))
    i, j = 1 + dx, 1 + dy
    labelbis[i : i + h, j : j + w] = label[:, :]

    return labelbis[1 : 1 + h, 1 : 1 + w]


def generatenoisyAIRS(noise, resolution):
    assert noise in ["nonoise", "hallucination", "pm1image", "pm1translation"]
    assert resolution in ["50cm", "1m"]

    if resolution == "50cm":
        root = "/scratchf/miniworld/christchurch/"
    else:
        root = "/scratchf/miniworld_1M/christchurch/"

    if noise == "nonoise":
        os.system("cp -r " + root + " build")
        return

    os.system("rm -r build/christchurch")
    os.system("mkdir build/christchurch")
    os.system("mkdir build/christchurch/train")
    os.system("cp -r " + root + "test build/christchurch")

    NB = 0
    while os.path.exists(root + "train/" + str(NB) + "_x.png"):
        NB += 1
    if NB == 0:
        print("wrong root path")
        quit()

    for i in range(NB):
        cmd = str(i) + "_x.png build/christchurch/train/" + str(i) + "_x.png"
        os.system("cp " + root + "train/" + cmd)

        path = root + "train/" + str(i) + "_y.png"
        label = PIL.Image.open(path).convert("L").copy()

        label = numpy.uint8(numpy.asarray(label))
        label = torch.Tensor(label)
        label = (label != 0).long()

        if noise == "hallucination":
            label = hallucination(label)
        if noise == "pm1image":
            label = pm1image(label)
        if noise == "pm1translation":
            label = pm1translation(label)

        label = (label != 0).numpy() * 254
        label = PIL.Image.fromarray(numpy.uint8(label))
        label.save("build/christchurch/train/" + str(i) + "_y.png")


if __name__ == "__main__":
    generatenoisyAIRS(sys.argv[1], sys.argv[2])
