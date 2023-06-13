
import os
import json

class FLAIR2:
    def __init__(self,root="/scratchf/CHALLENGE_IGN/FLAIR_2/"):
        with open(root+"flair-2_centroids_sp_to_patch.json") as fichier:
            self.coords = json.load(fichier)
        
        self.allImages = {}

        trainfolder = os.listdir(root+"flair_aerial_train/")
        testfolder = os.listdir(root+"flair_2_aerial_test/")
        assert not set(testfolder).intersection(set(trainfolder))
        
        print(testfolder)
        self.root = root

tmp = FLAIR2()
quit()

import queue
import threading
import PIL
from PIL import Image
import rasterio
import random


class CropExtractorDigitanie(threading.Thread):
    def __init__(self, paths, flag, tile):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 10
        self.tilesize = tile

        assert flag in ["even", "odd", "all"]
        self.flag = flag

        if self.flag == "even":
            paths = paths[::2]
        if self.flag == "odd":
            paths = paths[1::2]
        self.paths = paths

    def getImageAndLabel(self, i, torchformat=True):
        assert i < len(self.paths)

        with rasterio.open(self.paths[i][0]) as src:
            r = numpy.clip(src.read(1) * 2, 0, 1)
            g = numpy.clip(src.read(2) * 2, 0, 1)
            b = numpy.clip(src.read(3) * 2, 0, 1)
            x = numpy.stack([r, g, b], axis=0) * 255

        y = PIL.Image.open(self.paths[i][1]).convert("RGB").copy()
        y = numpy.asarray(y)
        y = numpy.uint8((y[:, :, 0] == 250) * (y[:, :, 1] == 50) * (y[:, :, 2] == 50))

        if y.shape != (2048, 2048):
            y, x = y[0:2048, 0:2048], x[:, 0:2048, 0:2048]

        if torchformat:
            return torch.Tensor(x), torch.Tensor(y)
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize=3):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def symetrie(self, x, y):
        if random.random() > 0.5:
            x[0] = numpy.transpose(x[0], axes=(1, 0))
            x[1] = numpy.transpose(x[1], axes=(1, 0))
            x[2] = numpy.transpose(x[2], axes=(1, 0))
            y = numpy.transpose(y, axes=(1, 0))
        if random.random() > 0.5:
            x[0] = numpy.flip(x[0], axis=1)
            x[1] = numpy.flip(x[1], axis=1)
            x[2] = numpy.flip(x[2], axis=1)
            y = numpy.flip(y, axis=1)
        if random.random() > 0.5:
            x[0] = numpy.flip(x[0], axis=0)
            x[1] = numpy.flip(x[1], axis=0)
            x[2] = numpy.flip(x[2], axis=0)
            y = numpy.flip(y, axis=0)
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
