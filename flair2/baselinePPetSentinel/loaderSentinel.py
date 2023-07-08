import torch
import dataloader


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv6 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.classif = torch.nn.Conv2d(200, 13, kernel_size=1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, s):
        s = self.lrelu(self.conv1(s))
        s = self.lrelu(self.conv2(s))
        s = self.lrelu(self.conv3(s))
        s = self.lrelu(self.conv4(s))
        s = self.lrelu(self.conv5(s))
        s = self.lrelu(self.conv6(s))

        p = self.classif(s)
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
        return p


class Sentinel(torch.nn.Module):
    def __init__(self):
        super(Sentinel, self).__init__()
        self.conv1 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.conv6 = torch.nn.Conv2d(200, 200, kernel_size=1)
        self.classif = torch.nn.Conv2d(200, 13, kernel_size=1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, s):
        s = self.lrelu(self.conv1(s))
        s = self.lrelu(self.conv2(s))
        s = self.lrelu(self.conv3(s))
        s = self.lrelu(self.conv4(s))
        s = self.lrelu(self.conv5(s))
        s = self.lrelu(self.conv6(s))

        p = self.classif(s)
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
        return p, s


if __name__ == "__main__":
    net = torch.load("../sentinelOnly/build/model.pth")

    copynet = dataloader.Sentinel()
    check = set()
    for module in net._modules:
        print(module)
        check.add(module)
        copynet._modules[module] = net._modules[module]

    for module in copynet._modules:
        check.remove(module)
    print("len(check)==", len(check))

    torch.save(copynet, "build/modelS.pth")
