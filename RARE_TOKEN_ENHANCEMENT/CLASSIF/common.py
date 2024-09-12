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


def confmat(gt, pred, classes=10):
    trick = classes * gt + pred
    mat = torch.bincount(trick, minlength=classes * classes)
    return mat.reshape(classes, classes)


import torchvision


def getEfficientNet():
    w = torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
    net = torchvision.models.efficientnet_b5(weights=w).features[0:2]
    for module in net.modules():
        if hasattr(module, "padding"):
            module.padding_mode = "reflect"
    net.add_module("pool", torch.nn.MaxPool2d(kernel_size=4))
    return net.eval()


def getEfficientNetV2():
    w = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    net = torchvision.models.efficientnet_v2_m(weights=w).features[0:2]
    for module in net.modules():
        if hasattr(module, "padding"):
            module.padding_mode = "reflect"
    net = net.eval()
    net.add_module("pool", torch.nn.MaxPool2d(kernel_size=4))
    return net.eval()


class MyNet(torch.nn.Module):
    def __init__(self, encoder):
        super(MyNet, self).__init__()
        self.encoder = encoder
        self.embed = torch.nn.Conv2d(24, 32, kernel_size=1)
        self.proj = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2)

        self.final1 = torch.nn.Conv2d(32, 64, kernel_size=1)
        self.final2 = torch.nn.Conv2d(64, 128, kernel_size=1)
        self.final3 = torch.nn.Linear(128, 10)

    def embedding(self, x):
        x = self.coding(x)
        return self.proj(x).flatten(2)

    def coding(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.embed(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        return x

    def forward(self, x):
        x = self.coding(x)

        x = self.final1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = self.final2(x)
        x, _ = x.flatten(2).max(2)
        return self.final3(x)


def getEurosat(raw=False):
    Tr, Pa = False, "build/2750/"
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if raw:
        tmp = [
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    else:
        tmp = [
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomVerticalFlip(0.5),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    aug = torchvision.transforms.Compose(tmp)
    return torchvision.datasets.ImageFolder(root=Pa, transform=aug)


class EurosatSplit(torch.utils.data.Dataset):
    def __init__(self, flag):
        assert flag in ["train", "test"]
        self.alldata = getEurosat(flag == "test")

        tmp = len(self.alldata)
        self.I = list(range(0, tmp, 20))
        if flag == "test":
            inv = set(self.I)
            self.I = [i for i in range(tmp) if i not in inv]
        self.size = len(self.I)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.alldata.__getitem__(self.I[idx])
