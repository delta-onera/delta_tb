import numpy
import PIL
from PIL import Image
import random
import os


def random_geometric_deformation(path):
    image = PIL.Image.open(path).convert("RGB").copy()

    M = numpy.zeros((3, 3))
    for i in range(2):
        for j in range(2):
            M[i][j] = random.uniform(-0.3, 0.3)
        M[i][i] += 1
        M[i][-1] = random.uniform(-30, 30)
    M[-1][-1] = 1

    image = image.transform(
        image.size, Image.AFFINE, data=M.flatten()[:6], resample=Image.BICUBIC
    )
    image = numpy.asarray(image)
    h, w, _ = image.shape
    h, w = h // 2, w // 2
    image = image[h - 128 : h + 128, w - 128 : w + 128, :]
    
    tmp = numpy.asarray([h, w, 1])
    print(tmp)
    print(numpy.dot(M, tmp))

    tmp = numpy.asarray([h - 128, w - 128])
    tmp = numpy.dot(M[:2, :2], tmp)
    M[0][-1] += tmp[0]
    M[1][-1] += tmp[1]

    tmp = numpy.asarray([128, 128, 1])
    print(tmp)
    print(numpy.dot(M, tmp))

    return numpy.uint8(image), M


path = "/scratchf/OSCD/rennes/pair/img1.png"
os.system("cp " + path + " build")
deformed_img, M = random_geometric_deformation(path)
deformed_img[128 - 3 : 128 + 3, 128 - 3 : 128 + 3, :] = 0
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test1.png")

q = numpy.asarray([128, 128, 1])
print(q)
q = numpy.dot(M, q)
print(q)

deformed_img, M = random_geometric_deformation(path)
q = numpy.dot(numpy.linalg.inv(M), q)
print(q)
qx, qy = int(q[0]), int(q[1])
deformed_img[qx - 3 : qx + 3, qy - 3 : qy + 3, :] = 0
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test2.png")

quit()

deformed_img, M2 = random_geometric_deformation("/scratchf/OSCD/rennes/pair/img1.png")
q = numpy.asarray([128, 128, 1])
q = numpy.dot(numpy.linalg.inv(M1), q)
q[2] = 1
q = numpy.dot(M2, q)
deformed_img[int(q[0]) - 3 : int(q[0]) + 3, int(q[1]) - 3 : int(q[1]) + 3, :] = 0
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test2.png")


quit()


import numpy
import PIL
from PIL import Image
import random


# Random roll, pitch, yaw rotations, translation, zoom
roll = random.uniform(-10, 10)
pitch = random.uniform(-10, 10)
yaw = random.uniform(-40, 40)
tx = random.uniform(-40, 40)
ty = random.uniform(-40, 40)
zoom = random.uniform(0.9, 1.1)

# Translation matrix
T = numpy.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

# Rotation matrices
Rx = numpy.array(
    [
        [1, 0, 0],
        [0, numpy.cos(numpy.radians(pitch)), -numpy.sin(numpy.radians(pitch))],
        [0, numpy.sin(numpy.radians(pitch)), numpy.cos(numpy.radians(pitch))],
    ]
)
Ry = numpy.array(
    [
        [numpy.cos(numpy.radians(yaw)), 0, -numpy.sin(numpy.radians(yaw))],
        [0, 1, 0],
        [numpy.sin(numpy.radians(yaw)), 0, numpy.cos(numpy.radians(yaw))],
    ]
)
Rz = numpy.array(
    [
        [numpy.cos(numpy.radians(roll)), -numpy.sin(numpy.radians(roll)), 0],
        [numpy.sin(numpy.radians(roll)), numpy.cos(numpy.radians(roll)), 0],
        [0, 0, 1],
    ]
)
R = numpy.matmul(Rz, numpy.matmul(Ry, Rx))

# Zoom matrix
Z = numpy.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, 1]])

# Transformation matrix
M = numpy.matmul(Z, numpy.matmul(R, T))

print(Z)
print(R)
print(T)
print(M)
quit()


def random_geometric_deformation(path):
    image = PIL.Image.open(path).convert("RGB").copy()

    # Random roll, pitch, yaw rotations, translation, zoom
    roll = random.uniform(-10, 10)
    pitch = random.uniform(-10, 10)
    yaw = random.uniform(-40, 40)
    tx = random.uniform(-40, 40)
    ty = random.uniform(-40, 40)
    zoom = random.uniform(0.9, 1.1)

    # Translation matrix
    T = numpy.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Rotation matrices
    Rx = numpy.array(
        [
            [1, 0, 0],
            [0, numpy.cos(numpy.radians(pitch)), -numpy.sin(numpy.radians(pitch))],
            [0, numpy.sin(numpy.radians(pitch)), numpy.cos(numpy.radians(pitch))],
        ]
    )
    Ry = numpy.array(
        [
            [numpy.cos(numpy.radians(yaw)), 0, -numpy.sin(numpy.radians(yaw))],
            [0, 1, 0],
            [numpy.sin(numpy.radians(yaw)), 0, numpy.cos(numpy.radians(yaw))],
        ]
    )
    Rz = numpy.array(
        [
            [numpy.cos(numpy.radians(roll)), -numpy.sin(numpy.radians(roll)), 0],
            [numpy.sin(numpy.radians(roll)), numpy.cos(numpy.radians(roll)), 0],
            [0, 0, 1],
        ]
    )
    R = numpy.matmul(Rz, numpy.matmul(Ry, Rx))

    # Zoom matrix
    Z = numpy.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, 1]])

    # Transformation matrix
    M = numpy.matmul(Z, numpy.matmul(R, T))

    # Apply the transformation to the image
    result_image = image.transform(
        image.size, Image.AFFINE, data=M.flatten()[:6], resample=Image.BICUBIC
    )

    # Compute the transformation matrices for going from new to old pixels and vice versa
    center_x = image.width / 2
    center_y = image.height / 2
    M_inv = numpy.linalg.inv(M)
    M_to_new_pixel = numpy.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])
    M_to_old_pixel = numpy.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])
    M_to_new_pixel = numpy.matmul(M_to_new_pixel, M_inv)
    M_to_old_pixel = numpy.matmul(M, M_to_old_pixel)

    # Crop the resulting image to size 256x256 centered in the image
    crop_size = 256
    left = int((result_image.width - crop_size) / 2)
    top = int((result_image.height - crop_size) / 2)
    right = left + crop_size
    bottom = top + crop_size
    result_image = result_image.crop((left, top, right, bottom))

    # Return the resulting image and the transformation matrices
    return result_image, M_to_new_pixel, M_to_old_pixel


import PIL
from PIL import Image
import numpy


def random_geometric_deformation(path):
    image = PIL.Image.open(path).convert("RGB").copy()

    # Random roll, pitch, and yaw rotations
    roll = numpy.random.uniform(-10, 10)
    pitch = numpy.random.uniform(-10, 10)
    yaw = numpy.random.uniform(-40, 40)

    # Random translation
    x_offset = numpy.random.uniform(-25, 25)
    y_offset = numpy.random.uniform(-25, 25)

    # Random zoom
    zoom = numpy.random.uniform(0.9, 1.1)

    # Apply the transformations
    image = image.rotate(roll, resample=Image.BICUBIC, expand=True)
    image = image.rotate(pitch, resample=Image.BICUBIC, expand=True)
    image = image.rotate(yaw, resample=Image.BICUBIC, expand=True)
    width, height = image.size
    new_width = int(width * zoom)
    new_height = int(height * zoom)
    image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    image = image.crop((x_offset, y_offset, x_offset + width, y_offset + height))

    # Compute the transformation matrix
    matrix = numpy.array(
        [
            [numpy.cos(yaw), -numpy.sin(yaw), x_offset],
            [numpy.sin(yaw), numpy.cos(yaw), y_offset],
            [0, 0, 1],
        ]
    )
    matrix = numpy.dot(
        numpy.array([[1, 0, 0], [0, 1, 0], [-new_width / 2, -new_height / 2, 1]]),
        matrix,
    )
    matrix = numpy.dot(
        numpy.array([[1 / zoom, 0, 0], [0, 1 / zoom, 0], [0, 0, 1]]), matrix
    )
    matrix = numpy.dot(
        numpy.array(
            [
                [numpy.cos(-pitch), 0, numpy.sin(-pitch)],
                [0, 1, 0],
                [-numpy.sin(-pitch), 0, numpy.cos(-pitch)],
            ]
        ),
        matrix,
    )
    matrix = numpy.dot(
        numpy.array(
            [
                [1, 0, 0],
                [0, numpy.cos(-roll), -numpy.sin(-roll)],
                [0, numpy.sin(-roll), numpy.cos(-roll)],
            ]
        ),
        matrix,
    )
    matrix = numpy.dot(
        numpy.array(
            [[1, 0, 0], [0, 1, 0], [new_width / 2 - 128, new_height / 2 - 128, 1]]
        ),
        matrix,
    )

    left = (width - 256) // 2
    top = (height - 256) // 2
    right = (width + 256) // 2
    bottom = (height + 256) // 2
    image = image.crop((left, top, right, bottom))

    return numpy.uint8(numpy.asarray(image)), matrix


deformed_img, M1 = random_geometric_deformation("/scratchf/OSCD/rennes/pair/img1.png")
deformed_img[128 - 3 : 128 + 3, 128 - 3 : 128 + 3, :] = 0
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test1.png")

deformed_img, M2 = random_geometric_deformation("/scratchf/OSCD/rennes/pair/img1.png")
q = numpy.asarray([128, 128, 1])
q = numpy.dot(numpy.linalg.inv(M1), q)
q[2] = 1
q = numpy.dot(M2, q)
deformed_img[int(q[0]) - 3 : int(q[0]) + 3, int(q[1]) - 3 : int(q[1]) + 3, :] = 0
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test2.png")


quit()


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


deformed_img, M1, _ = random_deformation("/scratchf/OSCD/rennes/pair/img1.png")
deformed_img[128 - 3 : 128 + 3, 128 - 3 : 128 + 3, :] = 0
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test1.png")

deformed_img, M2, _ = random_deformation("/scratchf/OSCD/rennes/pair/img1.png")
q = numpy.asarray([128, 128, 1])
q = numpy.dot(numpy.linalg.inv(M1), q)
q[2] = 1
q = numpy.dot(M2, q)
deformed_img[int(q[0]) - 3 : int(q[0]) + 3, int(q[1]) - 3 : int(q[1]) + 3, :] = 0
visu = PIL.Image.fromarray(deformed_img)
visu.save("build/test2.png")


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
