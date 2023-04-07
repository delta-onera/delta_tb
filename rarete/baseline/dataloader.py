import random
import PIL
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
from PIL import ImageDraw
import math
import numpy


def random_deformation(path, finalsize=256):
    translation_range = (0.1, 0.1)
    zoom_range = (0.9, 1.1)
    roll_range, pitch_range, yaw_range = 5, 5, 5

    img = PIL.Image.open(path).convert("RGB").copy()

    # Random roll, pitch, yaw
    roll = math.radians(random.uniform(-roll_range, roll_range))
    pitch = math.radians(random.uniform(-pitch_range, pitch_range))
    yaw = math.radians(random.uniform(-yaw_range, yaw_range))

    # Compute rotation matrix
    cx, cy = img.size[0] / 2, img.size[1] / 2
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

    # Random translation
    tx = random.uniform(-translation_range[0], translation_range[0]) * img.size[0]
    ty = random.uniform(-translation_range[1], translation_range[1]) * img.size[1]
    img = img.transform(
        img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BILINEAR
    )

    # Random zoom
    zoom = random.uniform(zoom_range[0], zoom_range[1])
    w, h = img.size
    nw, nh = int(w * zoom), int(h * zoom)
    img = img.resize((nw, nh), resample=Image.BILINEAR)

    # Crop the image to its final size
    left = (nw - finalsize) // 2
    top = (nh - finalsize) // 2
    right = (nw + finalsize) // 2
    bottom = (nh + finalsize) // 2
    img = img.crop((left, top, right, bottom))

    return numpy.uint8(numpy.asarray(img))


deformed_img = random_deformation("/scratchf/OSCD/rennes/pair/img1.png")

visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test.png")

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
