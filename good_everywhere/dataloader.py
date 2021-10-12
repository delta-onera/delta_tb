import os
import numpy as np
import PIL
from PIL import Image
import torch


def distancetransform(y, size=4):
    yy = 2.0 * y.unsqueeze(0) - 1
    yyy = torch.nn.functional.avg_pool2d(
        yy, kernel_size=2 * size + 1, stride=1, padding=size
    )
    D = 1.0 - 0.5 * (yy - yyy).abs()
    return D[0]


def symetrie(x, y, i, j, k):
    if i == 1:
        x, y = np.transpose(x, axes=(1, 0, 2)), np.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    if k == 1:
        x, y = np.flip(x, axis=1), np.flip(y, axis=1)
    return x.copy(), y.copy()


class MiniWorld:
    def __init__(self, custom=None, flag=None):
        assert custom is not None or flag is not None

        if custom is not None:
            self.cities = custom
        else:
            assert flag in ["train", "test"]
            self.cities = [
                "potsdam",
                "christchurch",
                "toulouse",
                "austin",
                "chicago",
                "kitsap",
                "tyrol-w",
                "vienna",
                "bruges",
                "Arlington",
                "Austin",
                "DC",
                "NewYork",
                "SanFrancisco",
                "Atlanta",
                "NewHaven",
                "Norfolk",
                "Seekonk",
            ]
            self.cities = [city + "/" + flag for city in self.cities]

        whereIam = os.uname()[1]
        if whereIam == "wdtim719z":
            self.root = "/data/miniworld/"
        if whereIam == "ldtis706z":
            self.root = "/media/achanhon/bigdata/data/miniworld/"
        if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
            self.root = "/scratch_ai4geo/miniworld/"

        self.nbImages = {}
        for city in self.cities:
            self.nbImages[city] = 0
            path = self.root + "/" + city + "/"
            while os.path.exists(path + str(self.nbImages) + "_x.png"):
                self.nbImages[city] += 1
            if self.nbImages[city] == 0:
                print("wrong path", path)
                quit()
        tot = sum([self.nbImages[city] for city in self.cities])
        print("found #cities, #image = ", len(self.cities), tot)

    def getImageAndLabel(self, city, i, torchFormat=False):
        assert city in self.cities
        assert i < self.nbImages[city]

        path = self.root + "/" + city + "/"
        image = PIL.Image.open(path + str(i) + "_x.png").convert("RGB").copy()
        image = np.uint8(np.asarray(image))

        label = PIL.Image.open(path + str(i) + "_y.png").convert("L").copy()
        label = np.uint8(np.asarray(label))
        label = np.uint8(label != 0)

        if torchFormat:
            x = torch.Tensor(numpy.transpose(image, axes=(2, 0, 1)))
            return x.unsqueeze(0), torch.Tensor(label)
        else:
            return image, label

    def getbatch(self, batchsize, priority=None):
        if priority is None:
            nb = batchsize // 2 // len(self.city) + 1
            priority = set([(city, nb) for city in self.cities])

        XY = []
        for city in self.cities:
            for i in range(priority[city]):
                XY.append(self.privategetone(city))
        X, Y = [x for x, _ in XY], [y for _, y in XY]
        X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
        return X, Y

    def openpytorchloader(self, tilesize=128, nbtiles=100000):
        self.tilesize = tilesize
        self.nbtiles = nbtiles // len(self.cities) + 1
        self.torchloader, self.iterator = {}, {}
        for city in self.cities:
            self.torchloader[city] = privatedataloader(self, city)
            self.iterator[city] = iter(self.torchloader[city])

    def privatedataloader(self, city):
        assert city in self.cities
        tile = self.tilesize
        tilesperimage = self.nbtiles // self.nbImages[city] + 1

        # crop
        XY = []
        for i in range(self.nbImages[city]):
            image, label = self.getImageAndLabel(city, i)

            row = np.random.randint(0, image.shape[0] - tile - 2, size=tilesperimage)
            col = np.random.randint(0, image.shape[1] - tile - 2, size=tilesperimage)

            for i in range(nbtilesperimage):
                im = image[row[i] : row[i] + tile, col[i] : col[i] + tile, :]
                mask = label[row[i] : row[i] + tile, col[i] : col[i] + tile]
                XY.append((im.copy(), mask.copy()))

        # symetrie
        symetrieflag = np.random.randint(0, 2, size=(len(XY), 3))
        XY = [
            (symetrie(x, y, symetrieflag[i][0], symetrieflag[i][1], symetrieflag[i][2]))
            for i, (x, y) in enumerate(XY)
        ]

        # pytorch
        X = torch.stack(
            [torch.Tensor(np.transpose(x, axes=(2, 0, 1))).cpu() for x, y in XY]
        )
        Y = torch.stack([torch.from_numpy(y).long().cpu() for x, y in XY])

        dataset = torch.utils.data.TensorDataset(X, Y)
        self.torchloader[city] = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=1
        )

    def privategetone(self, city):
        assert city in self.cities
        try:
            return next(self.iterator[city])
        except StopIteration:
            self.torchloader[city] = privatedataloader(self, city)
            self.iterator[city] = iter(self.torchloader[city])
            return next(self.iterator[city])
