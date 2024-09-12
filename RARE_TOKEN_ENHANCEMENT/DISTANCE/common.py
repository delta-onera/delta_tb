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


def dice(y, z):
    eps = 0.00001
    z = z.softmax(dim=1)
    z0, z1 = z[0, :, :], z[1, :, :]
    y0, y1 = (y == 0).float(), (y == 1).float()

    inter0, inter1 = (y0 * z0).sum(), (y1 * z1).sum()
    union0, union1 = (y0 + z1 * y0).sum(), (y1 + z0 * y1).sum()
    iou0, iou1 = (inter0 + eps) / (union0 + eps), (inter1 + eps) / (union1 + eps)

    if iou0 > iou1:
        iou = (2 * iou1 + iou0) / 3
    else:
        iou = (iou1 + 2 * iou0) / 3
    return 1 - iou


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


class MyNet(torch.nn.Module):
    def __init__(self, encoder):
        super(MyNet, self).__init__()
        self.encoder = encoder
        self.embed = torch.nn.Conv2d(24, 32, kernel_size=1)
        self.proj = torch.nn.Conv2d(32, 32, kernel_size=4, stride=4)
        self.semantic1 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.semantic2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.semantic3 = torch.nn.Conv2d(128, 64, kernel_size=1)
        self.semantic4 = torch.nn.Conv2d(128, 2, kernel_size=1)

    def embedding(self, x):
        x = self.coding(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        return (self.proj(x)[0]).flatten(1)

    def coding(self, x):
        _, H, W = x.shape
        x = x.view(1, 3, H, W)
        with torch.no_grad():
            x = self.encoder(x)
        x = self.embed(x)
        return x

    def forward(self, x1, x2):
        _, H, W = x1.shape
        x1 = self.coding(x1)
        x2 = self.coding(x2)

        x = torch.nn.functional.leaky_relu(self.semantic1(x1 - x2))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = torch.nn.functional.leaky_relu(self.semantic2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = self.semantic3(x)

        _, _, H_, W_ = x1.shape
        x = torch.nn.functional.interpolate(x, (H_, W_), mode="bilinear")
        x = torch.cat([x, x1, x2], dim=1)
        x = self.semantic4(x)
        x = torch.nn.functional.interpolate(x, (H, W), mode="bilinear")
        return x[0]


import os


def validCity(path):
    tmp = os.listdir(path)
    tmp = set(tmp)
    if "pair" not in tmp:
        return False
    if "cm" not in tmp:
        return False
    if os.path.isfile(path + "/pair") or os.path.isfile(path + "/cm"):
        return False
    if not os.path.isfile(path + "/pair/img1.png"):
        return False
    if not os.path.isfile(path + "/pair/img2.png"):
        return False
    if not os.path.isfile(path + "/cm/cm.png"):
        return False
    return True


class OSCD:
    def __init__(self, flag):
        assert flag in ["train", "test", "all"]
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        ROOT = "/scratchf/OSCD/"
        l = os.listdir(ROOT)
        l = [name for name in l if not os.path.isfile(ROOT + name)]
        l = sorted([name for name in l if validCity(ROOT + name)])

        if flag == "train":
            l = [l[i] for i in range(len(l)) if i % 4 == 0]
        if flag == "test":
            l = [l[i] for i in range(len(l)) if i % 4 != 0]
        self.ROOT, self.l = ROOT, l
        print(self.l)

        self.NBITER = 2000

    def get(self, i):
        path = self.ROOT + "/" + self.l[i] + "/"
        path1 = path + "pair/img1.png"
        path2 = path + "pair/img2.png"
        cm = path + "cm/cm.png"
        img1 = torchvision.io.read_image(path1)
        img2 = torchvision.io.read_image(path2)
        cm = torchvision.io.read_image(cm)
        cm = (cm[0] > 125).float()
        H, W = cm.shape
        cm = cm.view(1, 1, H, W)
        cm = torch.nn.functional.max_pool2d(cm, kernel_size=7, stride=1, padding=3)
        cm = cm[0][0].long()
        img1, img2 = img1.float() / 255, img2.float() / 255
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return img1, img2, cm

    def getrand(self):
        return self.get(int(torch.rand(1) * len(self.l)))


class S2Looking:
    def __init__(self, flag):
        assert flag in ["train", "test", "all"]
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

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
        self.NBITER = 3500

    def get(self, i):
        radix, number = self.l[i // 4]
        path1 = self.ROOT + radix + "Image1/" + number
        path2 = self.ROOT + radix + "Image2/" + number
        cm = self.ROOT + radix + "label/" + number

        img1 = torchvision.io.read_image(path1)
        img2 = torchvision.io.read_image(path2)
        cm = torchvision.io.read_image(cm)
        cm = (cm.sum(0) > 125).float()
        cm = cm.view(1, 1024, 1024)

        if i % 4 == 0:
            img1 = img1[:, 0:512, 0:512]
            img2 = img2[:, 0:512, 0:512]
            cm = cm[:, 0:512, 0:512]
        if i % 4 == 1:
            img1 = img1[:, 0:512, 512:]
            img2 = img2[:, 0:512, 512:]
            cm = cm[:, 0:512, 512:]
        if i % 4 == 2:
            img1 = img1[:, 512:, 512:]
            img2 = img2[:, 512:, 512:]
            cm = cm[:, 512:, 512:]
        if i % 4 == 3:
            img1 = img1[:, 512:, 0:512]
            img2 = img2[:, 512:, 0:512]
            cm = cm[:, 512:, 0:512]

        cm = torch.nn.functional.max_pool2d(cm, kernel_size=7, stride=1, padding=3)
        cm = cm[0].long()
        img1, img2 = img1.float() / 255, img2.float() / 255
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return img1, img2, cm

    def getrand(self):
        return self.get(int(torch.rand(1) * len(self.l) * 4))


class LEVIR:
    def __init__(self, flag):
        assert flag in ["train", "test", "all"]
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

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
        self.NBITER = 3500

    def get(self, i):
        radix, number = self.l[i // 4]
        path1 = self.ROOT + radix + "A/" + number
        path2 = self.ROOT + radix + "B/" + number
        cm = self.ROOT + radix + "label/" + number

        img1 = torchvision.io.read_image(path1)
        img2 = torchvision.io.read_image(path2)
        cm = torchvision.io.read_image(cm)
        cm = (cm.sum(0) > 125).float()
        cm = cm.view(1, 1024, 1024)

        if i % 4 == 0:
            img1 = img1[:, 0:512, 0:512]
            img2 = img2[:, 0:512, 0:512]
            cm = cm[:, 0:512, 0:512]
        if i % 4 == 1:
            img1 = img1[:, 0:512, 512:]
            img2 = img2[:, 0:512, 512:]
            cm = cm[:, 0:512, 512:]
        if i % 4 == 2:
            img1 = img1[:, 512:, 512:]
            img2 = img2[:, 512:, 512:]
            cm = cm[:, 512:, 512:]
        if i % 4 == 3:
            img1 = img1[:, 512:, 0:512]
            img2 = img2[:, 512:, 0:512]
            cm = cm[:, 512:, 0:512]

        cm = torch.nn.functional.max_pool2d(cm, kernel_size=7, stride=1, padding=3)
        cm = cm[0].long()
        img1, img2 = img1.float() / 255, img2.float() / 255
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return img1, img2, cm

    def getrand(self):
        return self.get(int(torch.rand(1) * len(self.l) * 4))
