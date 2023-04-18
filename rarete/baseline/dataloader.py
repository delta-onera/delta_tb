import numpy
import PIL
from PIL import Image
import random
import torch
import random
import queue
import threading
import os
import torchvision


def randomtransform():
    M = numpy.zeros((3, 3))
    for i in range(2):
        for j in range(2):
            M[i][j] = random.uniform(-0.25, 0.25) * random.uniform(0.3, 1)
        M[i][i] += 1
        M[i][-1] = random.uniform(-25, 25) * random.uniform(0.3, 1)
    M[-1][-1] = 1
    return M


def myimagetransform(image, M):
    output = numpy.zeros(image.shape)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            q = numpy.asarray([row, col, 1])
            q = numpy.dot(M, q)
            qx, qy = int(q[0]), int(q[1])
            if (0 <= qx < image.shape[0]) and (0 <= qy < image.shape[1]):
                output[row][col] = image[qx][qy][:]
    return numpy.uint8(output)


def myimagetransformCHATGPT(image, M):
    output = numpy.zeros(image.shape, dtype=numpy.uint8)
    image_shape = image.shape
    rows, cols = numpy.indices(
        image_shape[:2]
    )  # Create arrays of row and column indices

    # Compute transformed indices for all pixels in the image
    qx = numpy.round(M[0, 0] * rows + M[0, 1] * cols + M[0, 2]).astype(int)
    qy = numpy.round(M[1, 0] * rows + M[1, 1] * cols + M[1, 2]).astype(int)

    # Check if transformed indices are within bounds of the image
    valid_indices = (
        (qx >= 0) & (qx < image_shape[0]) & (qy >= 0) & (qy < image_shape[1])
    )

    # Use NumPy indexing to efficiently access corresponding pixels in the image
    output[rows[valid_indices], cols[valid_indices]] = image[
        qx[valid_indices], qy[valid_indices]
    ]

    return output


def random_geometric(path):
    image = PIL.Image.open(path).convert("RGB").copy()
    M = randomtransform()
    image = myimagetransformCHATGPT(numpy.asarray(image), M)

    h, w, _ = image.shape
    h, w = h // 2, w // 2
    image = image[h - 128 : h + 128, w - 128 : w + 128, :]

    tmp = numpy.asarray([h, w, 1])
    tmp = numpy.asarray([h - 128, w - 128])
    tmp = numpy.dot(M[:2, :2], tmp)
    M[0][-1] += tmp[0]
    M[1][-1] += tmp[1]

    return numpy.uint8(image), M


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1))) / 255


class Dataloader(threading.Thread):
    def __init__(self, paths, maxsize=10, batchsize=7, tilesize=256):
        threading.Thread.__init__(self)
        self.isrunning = False

        self.maxsize = maxsize
        self.batchsize = batchsize
        self.paths = paths

    def getRawImages(self, i):
        img1 = PIL.Image.open(self.paths[i] + "/pair/img1.png").convert("RGB")
        img1 = numpy.uint8(numpy.asarray(img1.copy()))
        img2 = PIL.Image.open(self.paths[i] + "/pair/img2.png").convert("RGB")
        img2 = numpy.uint8(numpy.asarray(img2.copy()))
        return img1, img2

    def getImages(self, i):
        img1, M1 = random_geometric(self.paths[i] + "/pair/img1.png")
        img2, M2 = random_geometric(self.paths[i] + "/pair/img2.png")
        M = numpy.matmul(numpy.linalg.inv(M2), M1)
        return img1, img2, M

    def getBatch(self):
        assert self.isrunning
        return self.q.get(block=True)

    def largeoverlap(self, m12):
        q = numpy.asarray([128, 128, 1])
        q = numpy.dot(m12, q)
        q = [int(q[0]), int(q[1])]
        return (0 <= q[0] < 256) and (0 <= q[1] < 256)

    def run(self):
        assert not self.isrunning
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        batchsize = self.batchsize

        while True:
            I = (torch.rand(self.batchsize) * len(self.paths)).long()
            x1 = torch.zeros(batchsize, 3, 256, 256)
            x2 = torch.zeros(batchsize, 3, 256, 256)
            m12 = torch.zeros(batchsize, 3, 3)
            for i in range(self.batchsize):
                img1, img2, M = self.getImages(I[i])
                if not self.largeoverlap(M):
                    img1, img2, M = self.getImages(I[i])
                img1, img2 = pilTOtorch(img1), pilTOtorch(img2)
                x1[i], x2[i], m12[i] = img1, img2, torch.Tensor(M)
            self.q.put((x1, x2, m12), block=True)


def getstdtraindataloader():
    root = "/scratchf/OSCD/"
    ll = os.listdir(root)
    ll = [l for l in ll if os.path.exists(root + l + "/pair/img1.png")]
    lll = []
    for l in ll:
        tmp = PIL.Image.open(root + l + "/pair/img1.png").copy()
        if min(tmp.size) > 256:
            lll.append(l)
    ll = [lll[i] for i in range(len(lll)) if i % 3 != 0]
    paths = [root + l for l in ll]
    return Dataloader(paths)


def getstdtestdataloader():
    root = "/scratchf/OSCD/"
    ll = os.listdir(root)
    ll = [l for l in ll if os.path.exists(root + l + "/pair/img1.png")]
    lll = []
    for l in ll:
        tmp = PIL.Image.open(root + l + "/pair/img1.png").copy()
        if min(tmp.size) > 256:
            lll.append(l)
    ll = [lll[i] for i in range(len(lll)) if i % 3 == 0]
    paths = [root + l for l in ll]
    return Dataloader(paths)


class RINET(torch.nn.Module):
    def __init__(self):
        super(RINET, self).__init__()
        self.net = torchvision.models.efficientnet_v2_s(weights="DEFAULT")
        self.net = self.net.features
        del self.net[7], self.net[6], self.net[5]

        self.f_ = torch.nn.Conv2d(128, 128, kernel_size=1)
        self.p_ = torch.nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x):
        return self.net((x - 0.5) * 2)

    def f(self, x):
        return self.f_(self.forward(x))

    def p(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return self.p_(z)

    def bothFP(self, x):
        with torch.no_grad():
            z = self.forward(x)
            return self.f_(z), self.p_(z)


if __name__ == "__main__":
    dataset = getstdtraindataloader()
    dataset.start()
    x1, x2, m12 = dataset.getBatch()

    for i in range(7):
        x1[i, :, 128 - 3 : 128 + 3, 128 - 3 : 128 + 3] = 0
        q = numpy.asarray([128, 128, 1])
        q = numpy.dot(m12[i], q)
        q = [int(q[0]), int(q[1])]
        if (0 <= q[0] < 256) and (0 <= q[1] < 256):
            x2[i, :, q[0] - 3 : q[0] + 3, q[1] - 3 : q[1] + 3] = 0

    for i in range(8):
        torchvision.utils.save_image(x1[i], "build/" + str(i) + "_1.png")
        torchvision.utils.save_image(x2[i], "build/" + str(i) + "_2.png")

    os._exit(0)
