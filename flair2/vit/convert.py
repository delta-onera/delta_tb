import torch, torchvision


class MyNet4(torch.nn.Module):
    def __init__(self):
        super(MyNet4, self).__init__()
        tmp = torchvision.models.swin_s(weights="DEFAULT").features
        del tmp[6:]
        with torch.no_grad():
            old = tmp[0][0].weight / 2
            tmp[0][0] = torch.nn.Conv2d(6, 96, kernel_size=4, stride=4)
            tmp[0][0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))
        self.vit = tmp
        self.classiflow = torch.nn.Conv2d(384, 13, kernel_size=1)

        ks = (2, 1, 1)
        self.conv1 = torch.nn.Conv3d(10, 32, kernel_size=ks, stride=ks, padding=0)
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=ks, stride=ks, padding=0)
        self.conv3 = torch.nn.Conv3d(64, 92, kernel_size=(3, 3, 3))
        self.conv4 = torch.nn.Conv3d(92, 128, kernel_size=ks, stride=ks, padding=0)
        self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(256, 256, kernel_size=3)

        self.merge1 = torch.nn.Conv2d(640, 640, kernel_size=1)
        self.merge2 = torch.nn.Conv2d(1024, 640, kernel_size=1)
        self.merge3 = torch.nn.Conv2d(1024, 128, kernel_size=1)

        self.decod1 = torch.nn.Conv2d(608, 128, kernel_size=1)
        self.decod2 = torch.nn.Conv2d(608, 128, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(608, 128, kernel_size=3, padding=1)
        self.decod4 = torch.nn.Conv2d(608, 128, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(608, 13, kernel_size=1)

        self.compress = torch.nn.Conv2d(128, 2, kernel_size=1)
        self.expand = torch.nn.Conv2d(2, 64, kernel_size=1)
        self.expand2 = torch.nn.Conv2d(13, 64, kernel_size=1)
        self.generate1 = torch.nn.Conv2d(64, 128, kernel_size=1)
        self.generate2 = torch.nn.Conv2d(128, 32, kernel_size=1)
        self.generate3 = torch.nn.Conv2d(32, 10, kernel_size=1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forwardRGB(self, x):
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        xm = xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.25
        x = x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        hr = self.vit[0:2](x)
        x = self.vit[2:](hr)

        hr = torch.transpose(hr, 2, 3)
        hr = torch.transpose(hr, 1, 2)
        x = torch.transpose(x, 2, 3)
        x = torch.transpose(x, 1, 2)

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
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge2(xs))
        xs = torch.cat([x, xs], dim=1)
        xs = self.lrelu(self.merge3(xs)).float()

        f = torch.nn.functional.interpolate(xs, size=(128, 128), mode="bilinear")
        x = x.float()
        x = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        x, f = x.to(dtype=hr.dtype), f.to(dtype=hr.dtype)
        f = torch.cat([f, x, hr], dim=1)
        f = self.lrelu(self.decod1(f))
        f = torch.cat([f, x, hr], dim=1)
        f = self.lrelu(self.decod2(f))
        f = torch.cat([f, x, hr], dim=1)
        f = self.lrelu(self.decod3(f))
        f = torch.cat([f, x, hr], dim=1)
        f = self.lrelu(self.decod4(f))
        f = torch.cat([f, x, hr], dim=1)
        p = self.classif(f)

        return p, xs

    def forward(self, x, s, mode=1):
        assert 1 <= mode <= 4

        if mode == 1:
            plow, x, hr = self.forwardRGB(x)
            s = self.forwardSentinel(s)
            p, _ = self.forwardClassifier(x, hr, s)
            p = p.float()
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


class MyLittleNet2(torch.nn.Module):
    def __init__(self):
        super(MyLittleNet2, self).__init__()
        tmp = torchvision.models.swin_s(weights="DEFAULT").features
        del tmp[6:]
        with torch.no_grad():
            old = tmp[0][0].weight / 2
            tmp[0][0] = torch.nn.Conv2d(6, 96, kernel_size=4, stride=4)
            tmp[0][0].weight = torch.nn.Parameter(torch.cat([old, old], dim=1))
        self.vit = tmp
        self.classiflow = torch.nn.Conv2d(384, 13, kernel_size=1)

    def forward(self, x):
        xm = torch.ones(x.shape[0], 1, 512, 512).cuda()
        xm = xm.to(dtype=x.dtype)
        x = ((x / 255) - 0.5) / 0.25
        x = x.to(dtype=xm.dtype)
        x = torch.cat([x, xm], dim=1)

        hr = self.vit[0:2](x)
        x = self.vit[2:](hr)
        x = torch.transpose(x, 2, 3)
        x = torch.transpose(x, 1, 2)

        p = self.classiflow(x).float()
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")

        return p


f = torch.load("build/baseline.pth")
g = MyLittleNet2()

g.vit = f.vit
g.classiflow = f.classiflow

torch.save(g, "build/model_converted.pth")
