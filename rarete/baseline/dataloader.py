import random
import PIL
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
from PIL import ImageDraw
import math
import numpy
import skimage


def random_deformation(path, finalsize=256):
    translation_range = (0.05, 0.05)
    zoom_range = (0.9, 1.1)
    roll_range, pitch_range, yaw_range = 10, 10, 40

    img = PIL.Image.open(path).convert("RGB").copy()

    w, h = 250, 250
    imgtemoin = numpy.zeros((500, 500, 3))
    imgtemoin[w - 5 : w + 5, h - 5 : h + 5, 0] = 255
    imgtemoin[w + 25 : w + 35, h - 5 : h + 5, 1] = 255
    imgtemoin[w - 5 : w + 5, h + 25 : h + 35, 2] = 255
    imgtemoin = PIL.Image.fromarray(numpy.uint8(imgtemoin))

    # Random roll, pitch, yaw
    roll = math.radians(random.uniform(-roll_range, roll_range))
    pitch = math.radians(random.uniform(-pitch_range, pitch_range))
    yaw = math.radians(random.uniform(-yaw_range, yaw_range))

    # Compute rotation matrix
    cx, cy = w, h
    cos_roll, sin_roll = math.cos(roll), math.sin(roll)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    a = cos_yaw * cos_pitch
    b = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    c = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    d = sin_yaw * cos_pitch
    e = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    f = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    g = -sin_pitch
    h = cos_pitch * sin_roll
    i = cos_pitch * cos_roll

    # Apply rotation matrix
    img = img.transform(
        img.size,
        Image.AFFINE,
        (a, b, -cx * a - b * cy + cx, d, e, -cx * d - e * cy + cy, g, h, i),
        resample=Image.BILINEAR,
    )
    imgtemoin = imgtemoin.transform(
        imgtemoin.size,
        Image.AFFINE,
        (a, b, -cx * a - b * cy + cx, d, e, -cx * d - e * cy + cy, g, h, i),
        resample=Image.BILINEAR,
    )

    # Random translation
    tx = random.uniform(-translation_range[0], translation_range[0]) * img.size[0]
    ty = random.uniform(-translation_range[1], translation_range[1]) * img.size[1]
    img = img.transform(
        img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BILINEAR
    )
    imgtemoin = imgtemoin.transform(
        imgtemoin.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BILINEAR
    )

    # Random zoom
    zoom = random.uniform(zoom_range[0], zoom_range[1])
    w, h = img.size
    nw, nh = int(w * zoom), int(h * zoom)
    img = img.resize((nw, nh), resample=Image.BILINEAR)
    imgtemoin = imgtemoin.resize((nw, nh), resample=Image.BILINEAR)

    # Crop the image to its final size
    left = (nw - finalsize) // 2
    top = (nh - finalsize) // 2
    right = (nw + finalsize) // 2
    bottom = (nh + finalsize) // 2
    img = img.crop((left, top, right, bottom))
    imgtemoin = imgtemoin.crop((left, top, right, bottom))

    # Extract correspondance from img temoin
    imgtemoin = numpy.uint8(numpy.asarray(imgtemoin))
    maskR = numpy.uint8(imgtemoin[:, :, 0] > 200)
    maskG = numpy.uint8(imgtemoin[:, :, 1] > 200)
    maskB = numpy.uint8(imgtemoin[:, :, 2] > 200)

    label_img = skimage.measure.label(maskR)
    props = skimage.measure.regionprops(label_img)
    c = max(props, key=lambda x: x.area).centroid
    cx, cy = int(c[0]), int(c[1])

    label_img = skimage.measure.label(maskG)
    props = skimage.measure.regionprops(label_img)
    c = max(props, key=lambda x: x.area).centroid
    ex, ey = int(c[0]), int(c[1])

    label_img = skimage.measure.label(maskB)
    props = skimage.measure.regionprops(label_img)
    c = max(props, key=lambda x: x.area).centroid
    gx, gy = int(c[0]), int(c[1])

    # 250,250 -> cx,cy
    # 280,250 -> ex,ey
    # 250,280 -> gx,gy

    # Construct the transformation matrix
    A = numpy.array([[250, 280, 250], [250, 250, 280], [1, 1, 1]])
    B = numpy.array([[cx, ex, gx], [cy, ey, gy], [1, 1, 1]])
    M = numpy.dot(B, numpy.linalg.inv(A))

    return numpy.uint8(numpy.asarray(img)), M, imgtemoin


deformed_img, _, _ = random_deformation("/scratchf/OSCD/rennes/pair/img1.png")
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test1.png")


deformed_img, M, temoin = random_deformation("/scratchf/OSCD/rennes/pair/img1.png")
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test2.png")
visu = PIL.Image.fromarray(temoin)
visu.save("build/test3.png")

w, h = 250, 250
temoin = numpy.zeros((500, 500, 3))
temoin[w - 5 : w + 5, h - 5 : h + 5, 0] = 255
temoin[w + 25 : w + 35, h - 5 : h + 5, 1] = 255
temoin[w - 5 : w + 5, h + 25 : h + 35, 2] = 255

temoinbis = numpy.zeros((256, 256, 3))
for i in range(150, 350):
    for j in range(150, 350):
        q = numpy.array([i, j, 1])
        q = numpy.dot(M, q)
        if 0 <= q[0] and q[0] < 256 and 0 <= q[1] and q[1] < 256:
            temoinbis[int(q[0]), int(q[1]), :] = temoin[i, j, :]
visu = PIL.Image.fromarray(numpy.uint8(temoinbis))
visu.save("build/test4.png")


quit()

import os
import PIL
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
from PIL import ImageDraw
import numpy
import torch
import random
import queue
import threading


class Dataloader(threading.Thread):
    def __init__(self, paths, maxsize=10, batchsize=8, tilesize=256):
        threading.Thread.__init__(self)
        self.isrunning = False

        self.maxsize = maxsize
        self.batchsize = batchsize
        self.paths = paths

    def getImages(self, i):
        assert i < len(self.paths)

        img1 = PIL.Image.open(self.paths[i] + "_1.png").convert("RGB").copy()
        img1 = numpy.uint8(numpy.asarray(img1))
        img2 = PIL.Image.open(self.paths[i] + "_2.png").convert("RGB").copy()
        img2 = numpy.uint8(numpy.asarray(img2))

        return img1, img2

    def geometricdistribution(img):
        pass

    def random_deformation(
        img, rotation_range=10, translation_range=(0.1, 0.1), zoom_range=(0.9, 1.1)
    ):
        # Random rotation
        angle = random.uniform(-rotation_range, rotation_range)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

        # Random translation
        tx = random.uniform(-translation_range[0], translation_range[0]) * img.size[0]
        ty = random.uniform(-translation_range[1], translation_range[1]) * img.size[1]
        img = img.transform(
            img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BICUBIC
        )

        # Random zoom
        zoom = random.uniform(zoom_range[0], zoom_range[1])
        w, h = img.size
        nw, nh = int(w * zoom), int(h * zoom)
        img = img.resize((nw, nh), resample=Image.BICUBIC)

        # Crop the image to its original size
        left = (nw - w) // 2
        top = (nh - h) // 2
        right = (nw + w) // 2
        bottom = (nh + h) // 2
        img = img.crop((left, top, right, bottom))

        return img

    def pilTOtorch(x):
        return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))

    def getBatch(self):
        assert self.isrunning
        return self.q.get(block=True)

    def run(self):
        assert not self.isrunning
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        batchsize = self.batchsize

        while True:
            I = (torch.rand(self.batchsize) * len(self.paths)).long()
            flag = numpy.random.randint(0, 2, size=(self.batchsize, 3))
            batch = torch.zeros(batchsize, 6, 48, 48)
            for i in range(self.batchsize):
                img1, img2 = self.getImages(I[i], torchformat=False)
                img1, img2 = symetrie(img1, flag[i]), symetrie(img2, flag[i])
                img1, img2 = pilTOtorch(img1), pilTOtorch(img2)
                batch[i, 0:3], batch[i, 3:6] = img1, img2
            self.q.put(batch, block=True)


def getstdtraindataloader():
    root = "../preprocessing/build/"
    paths = [str(i) for i in range(2358)]
    paths = [paths[i] for i in range(len(paths)) if i % 4 < 2]
    paths = [root + path for path in paths]
    return Dataloader(paths)


def getstdtestdataloader():
    root = "../preprocessing/build/"
    paths = [str(i) for i in range(2358)]
    paths = [paths[i] for i in range(len(paths)) if i % 4 >= 2]
    paths = [root + path for path in paths]
    return Dataloader(paths)
