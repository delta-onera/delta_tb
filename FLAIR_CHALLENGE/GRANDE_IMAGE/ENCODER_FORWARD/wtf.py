import os
import rasterio
import torch
import torchvision
import numpy
import PIL


class FLAIRTEST:
    def __init__(self, root, rootold):
        self.root = root

        self.domaines = os.listdir(root)
        self.paths = []
        for domaine in self.domaines:
            names = os.listdir(root + domaine)
            backup = set(names)
            names = [name[0:-4] for name in names if ".tif" in name]
            names = [name for name in names if (name + ".npy") in backup]

            for name in names:
                self.paths.append((root + domaine + "/" + name, name))

        self.rootold = rootold

        self.oldpaths = {}
        level1 = os.listdir(rootold)
        for folder in level1:
            level2 = os.listdir(rootold + folder)

            for subfolder in level2:
                path = rootold + folder + "/" + subfolder
                level3 = os.listdir(path + "/img")
                level3 = [name[4:] for name in level3 if ".aux" not in name]
                level3 = [name[0:-4] for name in level3]

                for name in level3:
                    x = path + "/img/IMG_" + name + ".tif"
                    name = "PRED_" + name + ".tif"
                    self.oldpaths[name] = x

    def getImageAndLabel(self, i):
        with rasterio.open(self.paths[i][0] + ".tif") as src_img:
            x = src_img.read()
            x = numpy.clip(numpy.nan_to_num(x), 0, 255)
            rahh = src_img.transform

        return torch.Tensor(x), rahh, self.paths[i][1]

    def rankfromlist(self, l):
        l = list(set(l))
        l = sorted(l)
        out = {}
        for i in range(len(l)):
            out[l[i]] = i * 512
        return out

    def exportresults(self, i, pred, rahh):
        boxes = numpy.load(self.paths[i][0] + ".npy", allow_pickle=True)

        cols = [boxes[j][1].left for j in range(boxes.shape[0])]
        rows = [boxes[j][1].top for j in range(boxes.shape[0])]
        cols, rows = self.rankfromlist(cols), self.rankfromlist(rows)

        tmp = [cols[j] for j in cols]
        if max(tmp) + 512 != pred.shape[1]:
            print(max(tmp) + 512, pred.shape)
        tmp = [rows[j] for j in rows]
        if max(tmp) + 512 != pred.shape[0]:
            print(max(tmp) + 512, pred.shape)

        for j in range(boxes.shape[0]):
            name, top, left = boxes[j][0], boxes[j][1].top, boxes[j][1].left
            top, left = rows[top], cols[left]

            imageInOld = self.oldpaths["PRED_0" + str(name) + ".tif"]
            with rasterio.open(self.oldpaths["PRED_0" + str(name) + ".tif"]) as src_img:
                lol = src_img.bounds
                west = boxes[j][1].left
                south = boxes[j][1].bottom
                east = boxes[j][1].right
                north = boxes[j][1].top
                width = pred.shape[1]
                height = pred.shape[0]
                print(rahh.from_bounds(west, south, east, north, width, height))
                print(top, left)
                quit()


dataset = FLAIRTEST("/scratchf/flair_merged/test/", "/scratchf/CHALLENGE_IGN/test/")
for i in range(len(dataset.paths)):
    print(i, "/", len(dataset.paths))
    x, _, projection = dataset.getImageAndLabel(i)
    dataset.exportresults(i, numpy.zeros(x.shape), projection)
