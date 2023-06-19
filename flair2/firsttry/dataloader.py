import torch
import numpy
import queue
import threading
import rasterio
import random
from functools import lru_cache


@lru_cache
def readSEN(path):
    return numpy.load(path)


class FLAIR2(threading.Thread):
    def __init__(self, root="/scratchf/CHALLENGE_IGN/FLAIR_2/", flag="test"):
        threading.Thread.__init__(self)
        assert flag in ["train", "val", "trainval", "test"]
        self.root = root
        self.isrunning = False
        self.flag = flag
        if flag == "test":
            self.paths = torch.load(root + "alltestpaths.pth")
        else:
            self.paths = torch.load(root + "alltrainpaths.pth")
        
        tmp = sorted(self.paths.keys())
        if flag == "train":
            tmp = [k for (i, k) in enumerate(tmp) if i % 3 != 0]
        if flag == "val":
            tmp = [k for (i, k) in enumerate(tmp) if i % 3 == 0]
        self.paths = {k: self.paths[k] for k in tmp}

    def get_(self, i):
        with rasterio.open(self.root + self.paths[i]["image"]) as src:
            r = numpy.clip(src.read(1), 0, 255)
            g = numpy.clip(src.read(2), 0, 255)
            b = numpy.clip(src.read(3), 0, 255)
            i = numpy.clip(src.read(4), 0, 255)
            e = numpy.clip(src.read(5), 0, 255)
            x = numpy.stack([r, g, b, i, e], axis=0) * 255

        sentinel = readSEN(self.root + self.paths[i]["sen"])
        row, col = self.paths[i]["coord"]
        sen = sentinel[:, :, row : row + 40, col : col + 40]
        sen=self.compressSEN(sen)

        if self.flag != "test":
            with rasterio.open(self.root + self.paths[i]["label"]) as src:
                y = numpy.clip(src.read(1), 0, 13)
            return x, sen, y
        else:
            return x, sen

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize=32):
        x = torch.zeros(batchsize, 3, 512, 512)
        sen = torch.zeros(batchsize,30, 30, 40,40)
        y = torch.zeros(batchsize, 512, 512)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize
        I = list(range(len(self.paths)))

        while True:
            random.shuffle(I)
            for i in I:
                image, label = self.getImageAndLabel(i, torchformat=False)

                r = int(random.random() * 1500)
                c = int(random.random() * 1500)
                im = image[:, r : r + tilesize, c : c + tilesize]
                mask = label[r : r + tilesize, c : c + tilesize]

                if numpy.sum(numpy.int64(mask != 0)) == 0:
                    continue
                if numpy.sum(numpy.int64(mask == 0)) == 0:
                    continue

                x, y = self.symetrie(im.copy(), mask.copy())
                x, y = torch.Tensor(x.copy()), torch.Tensor(y.copy())
                self.q.put((x, y), block=True)
