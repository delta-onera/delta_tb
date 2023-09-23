import torch, torchvision


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
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

        self.compress = torch.nn.Conv2d(320, 2, kernel_size=1)
        self.expand = torch.nn.Conv2d(2, 64, kernel_size=1)
        self.expand2 = torch.nn.Conv2d(13, 64, kernel_size=1)
        self.generate1 = torch.nn.Conv2d(64, 128, kernel_size=1)
        self.generate2 = torch.nn.Conv2d(128, 32, kernel_size=1)
        self.generate3 = torch.nn.Conv2d(32, 10, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forwardRGB(self, x):
        x = ((x / 255) - 0.5) / 0.5
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[2](self.backbone[1](self.backbone[0](x)))
        x = self.backbone[5](self.backbone[4](self.backbone[3](hr)))
        plow = self.classiflow(x)
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
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge2(xs))
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge3(xs))

        f1 = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        f2 = torch.nn.functional.interpolate(xs, size=(128, 128), mode="bilinear")
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

    def forward(self, x, s, mode=1):
        assert 1 <= mode <= 4

        if mode == 1:
            plow, x, hr = self.forwardRGB(x)
            s = self.forwardSentinel(s)
            p, _ = self.forwardClassifier(x, hr, s)
            p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
            return p + 0.1 * plow

        if mode == 2:
            plow, _, _ = self.forwardRGB(x)
            return plow

        if mode >= 3:
            if mode == 3:
                with torch.no_grad():
                    plow, x, hr = self.forwardRGB(x)
            else:
                plow, x, hr = self.forwardRGB(x)
            p, xs = self.forwardClassifier(x, hr, self.forwardSentinel(s))

            xs = self.compress(xs)
            xs = torch.nn.functional.interpolate(xs, size=(40, 40), mode="bilinear")
            xs = self.expand(xs)
            ps = torch.nn.functional.interpolate(p, size=(40, 40), mode="bilinear")
            ps = self.expand2(ps)
            xs = ps * xs

            xs = self.lrelu(self.generate1(xs))
            xs = self.lrelu(self.generate2(xs))
            xs = self.generate3(xs)
            xs = xs * 0.1 + 0.9 * torch.clamp(xs, -1, 1)
            assert xs.shape[1:] == (10, 40, 40)

            loss = ((xs - s.mean(2)) ** 2).flatten().mean()
            tmp = (xs.unsqueeze(2) - s) ** 2
            assert tmp.shape[1:] == (10, 32, 40, 40)
            losses = torch.zeros(tmp.shape[0]).cuda()
            for i in range(tmp.shape[0]):
                losses[i] = min([tmp[i, :, j, :, :].mean() for j in range(32)])
            loss = loss + losses.mean()

            p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")
            return p + 0.1 * plow, loss


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

    def forward(self, x):
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


f = torch.load("build/baseline.pth")
g = MyLittleNet()

g.backbone = f.backbone
g.classiflow = f.classiflow

torch.save(g, "build/model_converted.pth")
