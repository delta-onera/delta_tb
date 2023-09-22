import torch
import dataloader
import numpy
import PIL
from PIL import Image
import time


class MyNet3(torch.nn.Module):
    def __init__(self):
        super(MyNet3, self).__init__()
        tmp = torchvision.models.regnet_y_16gf(
            weights=torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_V2
        )
        with torch.no_grad():
            old = tmp.stem[0].weight / 2
            tmp.stem[0] = torch.nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1)
            tmp.stem[0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))
        del tmp.trunk_output.block4
        del tmp.fc
        self.reg = tmp
        self.classiflow = torch.nn.Conv2d(1232, 13, kernel_size=1)

        ks = (2, 1, 1)
        self.conv1 = torch.nn.Conv3d(10, 32, kernel_size=ks, stride=ks, padding=0)
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=ks, stride=ks, padding=0)
        self.conv3 = torch.nn.Conv3d(64, 92, kernel_size=(3, 3, 3))
        self.conv4 = torch.nn.Conv3d(92, 128, kernel_size=ks, stride=ks, padding=0)
        self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(256, 280, kernel_size=3)

        self.merge1 = torch.nn.Conv2d(1512, 512, kernel_size=1)
        self.merge2 = torch.nn.Conv2d(2024, 512, kernel_size=1)
        self.merge3 = torch.nn.Conv2d(2024, 112, kernel_size=1)

        self.decod1 = torch.nn.Conv2d(336, 112, kernel_size=1)
        self.decod2 = torch.nn.Conv2d(448, 112, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(448, 112, kernel_size=3, padding=1)
        self.decod4 = torch.nn.Conv2d(448, 112, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(448, 13, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forwardRGB(self, x):
        xm = torch.ones(x.shape[0], 1, 512, 512).cuda()
        xm = xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.25
        x = x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        hr = self.reg.trunk_output.block1(self.reg.stem(x))  # 224
        x = self.reg.trunk_output.block3(self.reg.trunk_output.block2(hr))  # 1232
        plow = self.classiflow(x).float()
        plow = torch.nn.functional.interpolate(plow, size=(512, 512), mode="bilinear")

        return plow, x, hr

    def forwardSentinel(self, s):
        s = self.lrelu(self.conv1(s))
        s = self.lrelu(self.conv2(s))
        s = self.lrelu(self.conv3(s))
        s = self.lrelu(self.conv4(s))

        ss = s.mean(2)
        s, _ = s.max(2)
        s = torch.cat([s, ss], dim=1)

        s = self.lrelu(self.conv5(s))
        s = self.lrelu(self.conv6(s))
        s = self.lrelu(self.conv7(s))
        return s

    def forwardClassifier(self, x, hr, s):
        xs = torch.cat([x, s], dim=1)
        xs = self.lrelu(self.merge1(xs))
        xs = torch.cat([x, s, xs], dim=1)
        xs = self.lrelu(self.merge2(xs))
        xs = torch.cat([x, s, xs], dim=1)
        xs = self.lrelu(self.merge3(xs)).float()

        f = torch.nn.functional.interpolate(xs, size=(128, 128), mode="bilinear")
        f = f.to(dtype=hr.dtype)
        f2 = torch.cat([f, hr], dim=1)
        f2 = self.lrelu(self.decod1(f2))
        f2 = torch.cat([f, f2, hr], dim=1)
        f2 = self.lrelu(self.decod2(f2))
        f2 = torch.cat([f, f2, hr], dim=1)
        f2 = self.lrelu(self.decod3(f2))
        f2 = torch.cat([f, f2, hr], dim=1)
        f2 = self.lrelu(self.decod4(f2))
        f2 = torch.cat([f, f2, hr], dim=1)
        p = self.classif(f2)

        return p, xs

    def forward(self, x, s):
        plow, x, hr = self.forwardRGB(x)
        s = self.forwardSentinel(s)
        p, _ = self.forwardClassifier(x, hr, s)
        p = p.float()
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
        return p + 0.1 * plow


class MyNet2(torch.nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()
        tmp = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
        with torch.no_grad():
            old = tmp[0][0].weight / 2
            tmp[0][0] = torch.nn.Conv2d(
                6, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            tmp[0][0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))
        del tmp[7]
        del tmp[6]
        self.backbone = tmp
        self.classiflow = torch.nn.Conv2d(160, 13, kernel_size=1)

        ks = (2, 1, 1)
        self.conv1 = torch.nn.Conv3d(10, 32, kernel_size=ks, stride=ks, padding=0)
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=ks, stride=ks, padding=0)
        self.conv3 = torch.nn.Conv3d(64, 92, kernel_size=(3, 3, 3))
        self.conv4 = torch.nn.Conv3d(92, 128, kernel_size=ks, stride=ks, padding=0)
        self.conv5 = torch.nn.Conv2d(256, 160, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(160, 160, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(160, 160, kernel_size=3)

        self.merge1 = torch.nn.Conv2d(320, 512, kernel_size=1)
        self.merge2 = torch.nn.Conv2d(672, 768, kernel_size=1)
        self.merge3 = torch.nn.Conv2d(928, 160, kernel_size=1)

        self.decod1 = torch.nn.Conv2d(368, 208, kernel_size=1)
        self.decod2 = torch.nn.Conv2d(416, 208, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(416, 208, kernel_size=3, padding=1)
        self.decod4 = torch.nn.Conv2d(416, 304, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(512, 13, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.stride = []
        for i in range(4):
            for j in range(4):
                self.stride.append((i, j))

    def forwardRGB(self, x):
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        xm = xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.5
        x = x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[2](self.backbone[1](self.backbone[0](x)))
        x = self.backbone[5](self.backbone[4](self.backbone[3](hr)))
        plow = self.classiflow(x).float()
        plow = torch.nn.functional.interpolate(plow, size=(512, 512), mode="bilinear")

        return plow, x, hr

    def forwardSentinel(self, s):
        s = self.lrelu(self.conv1(s))
        s = self.lrelu(self.conv2(s))
        s = self.lrelu(self.conv3(s))
        s = self.lrelu(self.conv4(s))

        ss = s.mean(2)
        s, _ = s.max(2)
        s = torch.cat([s, ss], dim=1)

        s = self.lrelu(self.conv5(s))
        s = self.lrelu(self.conv6(s))
        s = self.lrelu(self.conv7(s))
        return s

    def expand4(self, x):
        B, ch, H, W = x.shape
        dt = x.dtype
        z = torch.zeros((B, ch, H * 4, W * 4), dtype=dt).cuda()
        for i, j in self.stride:
            z[:, :, i::4, j::4] = x
        return z

    def forwardClassifier(self, x, hr, s):
        xs = torch.cat([x, s], dim=1)
        xs = self.lrelu(self.merge1(xs))
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge2(xs))
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge3(xs))

        f1 = self.expand4(x)
        f2 = self.expand4(xs)
        f = torch.cat([f1, f2, hr], dim=1)
        f2 = self.lrelu(self.decod1(f))
        f = torch.cat([f1, f2, hr], dim=1)
        f2 = self.lrelu(self.decod2(f))
        f = torch.cat([f1, f2, hr], dim=1)
        f2 = self.lrelu(self.decod3(f))
        f = torch.cat([f1, f2, hr], dim=1)
        f2 = self.lrelu(self.decod4(f))
        f = torch.cat([f1, f2, hr], dim=1)
        p = self.classif(f)

        return p, torch.cat([x, xs], dim=1)

    def forward(self, x, s):
        plow, x, hr = self.forwardRGB(x)
        s = self.forwardSentinel(s)
        p, _ = self.forwardClassifier(x, hr, s)
        p = p.float()
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
        return p + 0.1 * plow


class MyLittleNet(torch.nn.Module):
    def __init__(self):
        super(MyLittleNet, self).__init__()
        tmp = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
        with torch.no_grad():
            old = tmp[0][0].weight / 2
            tmp[0][0] = torch.nn.Conv2d(
                6, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            tmp[0][0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))
        del tmp[7]
        del tmp[6]
        self.backbone = tmp
        self.classiflow = torch.nn.Conv2d(160, 13, kernel_size=1)

    def forward(self, x, s):
        xm = torch.ones(x.shape[0], 1, 512, 512).cuda()
        xm = xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.25
        x = x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[2](self.backbone[1](self.backbone[0](x)))
        x = self.backbone[5](self.backbone[4](self.backbone[3](hr)))
        p = self.classiflow(x).float()
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")

        return p


assert torch.cuda.is_available()
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True


def number6(i):
    s = str(i)
    while len(s) < 6:
        s = "0" + s
    return s


class DeepEnsemble(torch.nn.Module):
    def __init__(self, m1, m2, m3, m4, m5, m6):
        super(DeepEnsemble, self).__init__()
        self.m1 = torch.load(m1).half()
        self.m2 = torch.load(m2).half()
        self.m3 = torch.load(m3).half()
        self.m4 = torch.load(m4).half()
        self.m5 = torch.load(m5).half()
        self.m6 = torch.load(m6).half()

    def forward(self, x, s):
        p1 = self.m1(x, s)
        p2 = self.m2(x, s)
        p3 = self.m3(x, s)
        p4 = self.m4(x, s)
        p5 = self.m5(x, s)
        p6 = self.m6(x, s) * 0.5

        p = torch.stack([p1, p2, p3, p4, p5, p6], dim=0)
        # p = torch.stack([p1, p2, p3, p4], dim=0)
        pp = torch.nn.functional.relu(p)
        np = torch.nn.functional.relu(-p)
        np = np + np * np  # increase inhibition effect

        p = (pp - np).sum(0)

        p[:, 7, :, :] *= 1.125
        p[:, 9, :, :] *= 1.1
        p[:, 10, :, :] *= 1.1
        return p


T0 = time.time()
print("load model")
net = DeepEnsemble(
    "../semisup2/build/model.pth",
    "../fast/build/model_converted.pth",
    "../vit/build/model.pth",
    "../vitbis/build/model.pth",
    "../autrebackbonebis/build/model_converted.pth",
    "../semisup2bis/build/model_converted.pth",
)
######################### 63.07

net = net.cuda()
net.eval()
net.half()

print("load data")
dataset = dataloader.FLAIR2()
N = len(dataset.paths)
dataset.start()

writter = dataloader.ImageWritter(N)
writter.start()

print("test")
stats = torch.zeros((13, 13)).cuda()
with torch.no_grad():
    for iiii in range(N):
        name, x, s = dataset.asynchroneGet()
        x, s = x.half().cuda(), s.half().cuda()

        z = net(x.unsqueeze(0), s.unsqueeze(0))
        _, z = z[0].max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        writter.asynchronePush("build/PRED_" + number6(name) + ".tif", z)

print("almost done", time.time() - T0)
writter.join()
print("done", time.time() - T0)
