import torch


# return I,J such that
# X[:,J[k]] is the nearest neighbour of X[:,I[k]]
# and such that || X[:,I[k]]-X[:,J[k]] || is decreasing
def computematching(X):
    with torch.no_grad():
        F, N = X.shape
        D = X.view(F, N, 1) - X.view(F, 1, N)
        D = (D**2).sum(0)
        D, J = D.sort(1)
        D, J = D[:, 1], J[:, 1]
        _, I = D.sort(descending=True)
        J = J[I]
        return I, J


def getRarityLoss(X, k=10):
    # removing affine transformation
    norm2 = (X**2).sum(0)
    Xm = norm2.max()
    X = X / torch.sqrt(Xm + 0.01)

    # compute normalized distance
    I, J = computematching(X)
    I, J = I[0:k], J[0:k]
    rarity = ((X[:, I] - X[:, J]) ** 2).sum(0).mean()
    # we want rarity to be highest possible
    # yet, distance between points in unit disk is bounded by 4
    return 4.0 - rarity


import torchvision


def getEfficientNet():
    w = torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
    net = torchvision.models.efficientnet_b5(weights=w).features[0:2]
    for module in net.modules():
        if hasattr(module, "padding"):
            module.padding_mode = "reflect"
    return net.eval()


def getEfficientNetV2():
    w = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    net = torchvision.models.efficientnet_v2_m(weights=w).features[0:2]
    for module in net.modules():
        if hasattr(module, "padding"):
            module.padding_mode = "reflect"
    return net.eval()


def Id():
    return torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).cuda()


class MyNet(torch.nn.Module):
    def __init__(self, encoder):
        super(MyNet, self).__init__()
        self.encoder = encoder
        self.embed = torch.nn.Conv2d(24, 32, kernel_size=1)
        self.proj = torch.nn.Conv2d(32, 32, kernel_size=4, stride=4)

        self.dense = torch.nn.Conv2d(32, 8, kernel_size=5, padding=2)

        self.final1 = torch.nn.Conv2d(48, 64, kernel_size=1)
        self.final2 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.final3 = torch.nn.Linear(128, 6)

        self.DELTA = []
        for i in range(3):
            for j in range(3):
                self.DELTA.append((i, j))

    def embedding(self, x):
        x = self.coding(x)
        return (self.proj(x)[0]).flatten(1)

    def coding(self, x):
        _, H, W = x.shape
        x = x.view(1, 3, H, W)
        with torch.no_grad():
            x = self.encoder(x)
        x = self.embed(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        return x

    def pixelMatch_(self, x, y):
        _, C, H, W = x.shape
        diff = torch.zeros(9, H, W).cuda()
        for i in range(9):
            dh, dw = self.DELTA[i]
            yc = y[0, :, 1 : H - 1, 1 : W - 1]
            tmp = x[0, :, dh : dh + H - 2, dw : dw + W - 2] - yc
            diff[i, 1 : H - 1, 1 : W - 1] = (tmp * tmp).sum(0)

        diff = 1 / torch.sqrt(diff + 0.01)
        diff = diff / diff.sum(0).view(1, H, W)
        return diff.view(1, 9, H, W)

    def positional(self, shape):
        with torch.no_grad():
            _, _, H, W = shape
            h = torch.arange(H).float()
            w = torch.arange(W).float()
            h = 2.0 * h / h.max() - 1.0
            w = 2.0 * w / w.max() - 1.0

            p = torch.ones(1, 2, H, W).cuda()
            p[:, 0, :, :] *= h.view(1, H, 1).cuda()
            p[:, 1, :, :] *= w.view(1, 1, W).cuda()
            return p

    def pixelMatch(self, x, y):
        a = self.pixelMatch_(x, y)
        b = self.pixelMatch_(y, x)
        s = self.positional(x.shape)
        return torch.cat([a, b, s], dim=1)

    def forward(self, x1, x2):
        x1 = self.coding(x1)
        x2 = self.coding(x2)

        level1 = self.pixelMatch(x1, x2)
        level1 = torch.nn.functional.max_pool2d(level1, kernel_size=2)

        x1 = torch.nn.functional.max_pool2d(x1, kernel_size=2)
        x2 = torch.nn.functional.max_pool2d(x2, kernel_size=2)
        level2 = self.pixelMatch(x1, x2)

        dense = torch.nn.functional.leaky_relu(self.dense(x1 - x2))

        x = torch.cat([level1, level2, dense], dim=1)
        x = self.final1(x)
        x = torch.nn.functional.adaptive_max_pool2d(x, 3)
        x = self.final2(x).view(1, 128)
        tmp = (x * x).sum() + 0.001
        x = x / torch.sqrt(tmp).view(1, 1)
        x = self.final3(x).view(2, 3)
        x = torch.cat([x, torch.zeros(1, 3).cuda()], dim=0)
        return Id() + torch.clamp(x, -0.11, 0.11) + x * 0.01


class BadAffine:
    def __init__(self):
        CANONICAL_TRANSFORM = []
        CANONICAL_TRANSFORM.append(
            torch.Tensor(
                [
                    [0.99619855292, 0.08711167063, 0],
                    [-0.08711167063, 0.99619855292, 0],
                    [0, 0, 1],
                ],
            )
        )
        CANONICAL_TRANSFORM.append(torch.Tensor([[1, 0, 0.05], [0, 1, 0], [0, 0, 1]]))
        CANONICAL_TRANSFORM.append(torch.Tensor([[1, 0, 0], [0, 1, -0.05], [0, 0, 1]]))
        CANONICAL_TRANSFORM.append(torch.Tensor([[1.05, 0, 0], [0, 1, 0], [0, 0, 1]]))
        CANONICAL_TRANSFORM.append(torch.Tensor([[1, 0, 0], [0, 0.95, 0], [0, 0, 1]]))
        tmp = torch.matmul(CANONICAL_TRANSFORM[0], CANONICAL_TRANSFORM[1])
        tmp2 = torch.matmul(CANONICAL_TRANSFORM[2], CANONICAL_TRANSFORM[3])
        tmp3 = torch.matmul(tmp, tmp2)
        tmp4 = torch.matmul(tmp3, CANONICAL_TRANSFORM[4])
        CANONICAL_TRANSFORM.append(tmp)
        CANONICAL_TRANSFORM.append(tmp2)
        CANONICAL_TRANSFORM.append(tmp3)
        CANONICAL_TRANSFORM.append(tmp4)
        CANONICAL_TRANSFORM.append(Id().cpu())
        self.CANONICAL_TRANSFORM = CANONICAL_TRANSFORM

    def transform(self, x, A, s):
        with torch.no_grad():
            _, H, W = x.shape
            invA = torch.inverse(A)

            tmp = torch.arange(s).float()
            tmp = 2 * tmp / tmp.max() - 1.0
            pos = torch.ones(3, s, s)
            pos[0] *= tmp.view(s, 1)
            pos[1] *= tmp.view(1, s)
            pos[2] = 1

            pos = pos.flatten(1)
            q = torch.matmul(invA, pos).transpose(0, 1) * s / 2
            qr, qc = (q[:, 0] + H / 2).long(), (q[:, 1] + W / 2).long()

            z = x[:, qr, qc]
            return z.view(3, s, s)

    def random(self, x, x_, s=224):
        _, H, W = x.shape
        assert min(H, W) >= 320, str(H) + " " + str(W)
        with torch.no_grad():
            A = Id().cpu()
            A[0:2, :] += torch.rand(2, 3) * 0.2 - 0.1

            y = self.transform(x_, A, s)
            x = x[:, (H - s) // 2 : (H + s) // 2, (W - s) // 2 : (W + s) // 2]
            return x, y, A

    def notrandom(self, x, x_, i, s=224):
        _, H, W = x.shape
        assert min(H, W) >= 320
        with torch.no_grad():
            y = self.transform(x_, self.CANONICAL_TRANSFORM[i], s)
            x = x[:, (H - s) // 2 : (H + s) // 2, (W - s) // 2 : (W + s) // 2]
            return x, y, self.CANONICAL_TRANSFORM[i].clone()


import os


def validCity(path):
    tmp = os.listdir(path)
    tmp = set(tmp)
    if "pair" not in tmp:
        return False
    if os.path.isfile(path + "/pair"):
        return False
    if not os.path.isfile(path + "/pair/img1.png"):
        return False
    if not os.path.isfile(path + "/pair/img2.png"):
        return False
    return True


class OSCD:
    def __init__(self, flag):
        assert flag in ["train", "test", "all"]
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.badaffine = BadAffine()

        ROOT = "/scratchf/OSCD/"
        l = os.listdir(ROOT)
        l = [name for name in l if not os.path.isfile(ROOT + name)]
        # remove too small images
        l = [name for name in l if name not in ["norcia"]]
        l = sorted([name for name in l if validCity(ROOT + name)])

        if flag == "train":
            l = [l[i] for i in range(len(l)) if i % 4 == 0]
        if flag == "test":
            l = [l[i] for i in range(len(l)) if i % 4 != 0]
        self.ROOT, self.l = ROOT, l

    def get(self, i, j=None):
        path = self.ROOT + "/" + self.l[i] + "/"
        path1 = path + "pair/img1.png"
        path2 = path + "pair/img2.png"
        img1 = torchvision.io.read_image(path1)
        img2 = torchvision.io.read_image(path2)
        img1, img2 = img1.float() / 255, img2.float() / 255
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        if j is not None:
            return self.badaffine.notrandom(img1, img2, j)
        else:
            return self.badaffine.random(img1, img2)

    def getrand(self):
        return self.get(int(torch.rand(1) * len(self.l)))


class S2Looking:
    def __init__(self, flag):
        assert flag in ["train", "test", "all"]
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.badaffine = BadAffine()

        ROOT = "/scratchf/NAVISAR/S2Looking/"

        testfiles = os.listdir(ROOT + "train/Image1")
        testfiles = [("train/", n) for n in testfiles]
        trainfiles = os.listdir(ROOT + "val/Image1")
        trainfiles = [("val/", n) for n in trainfiles]
        if flag == "train":
            l = trainfiles
        if flag == "test":
            l = testfiles
        if flag == "all":
            l = trainfiles + testfiles
        l = [(r, n) for (r, n) in l if ".png" in n]

        self.ROOT, self.l = ROOT, l

    def resize(self, img):
        img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, 512, mode="bilinear")
        return img[0]

    def get(self, i, j=None):
        radix, number = self.l[i]
        path1 = self.ROOT + radix + "Image1/" + number
        path2 = self.ROOT + radix + "Image2/" + number

        img1 = torchvision.io.read_image(path1)
        img2 = torchvision.io.read_image(path2)
        img1, img2 = img1.float() / 255, img2.float() / 255
        img1 = self.resize(self.normalize(img1))
        img2 = self.resize(self.normalize(img2))

        if j is not None:
            return self.badaffine.notrandom(img1, img2, j)
        else:
            return self.badaffine.random(img1, img2)

    def getrand(self):
        return self.get(int(torch.rand(1) * len(self.l)))


class LEVIR:
    def __init__(self, flag):
        assert flag in ["train", "test", "all"]
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.badaffine = BadAffine()

        ROOT = "/scratchf/NAVISAR/LEVIR/LEVIR-CD+/"

        testfiles = os.listdir(ROOT + "train/A")
        testfiles = [("train/", n) for n in testfiles]
        trainfiles = os.listdir(ROOT + "test/A")
        trainfiles = [("test/", n) for n in trainfiles]
        if flag == "train":
            l = trainfiles
        if flag == "test":
            l = testfiles
        if flag == "all":
            l = trainfiles + testfiles
        l = [(r, n) for (r, n) in l if ".png" in n]

        self.ROOT, self.l = ROOT, l

    def resize(self, img):
        img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, 512, mode="bilinear")
        return img[0]

    def get(self, i, j=None):
        radix, number = self.l[i]
        path1 = self.ROOT + radix + "A/" + number
        path2 = self.ROOT + radix + "B/" + number

        img1 = torchvision.io.read_image(path1)
        img2 = torchvision.io.read_image(path2)
        img1, img2 = img1.float() / 255, img2.float() / 255
        img1 = self.resize(self.normalize(img1))
        img2 = self.resize(self.normalize(img2))

        if j is not None:
            return self.badaffine.notrandom(img1, img2, j)
        else:
            return self.badaffine.random(img1, img2)

    def getrand(self):
        return self.get(int(torch.rand(1) * len(self.l)))
