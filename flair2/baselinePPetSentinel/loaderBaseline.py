import torch
import torchvision


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

        self.decod1 = torch.nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.decod2 = torch.nn.Conv2d(224, 128, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(288, 256, kernel_size=1)
        self.decod4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.5
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[2](self.backbone[1](self.backbone[0](x)))
        x = self.backbone[5](self.backbone[4](self.backbone[3](hr)))
        plow = self.classiflow(x)
        plow = torch.nn.functional.interpolate(plow, size=(512, 512), mode="bilinear")

        x = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        hr = torch.nn.functional.gelu(self.decod1(hr))
        hr = torch.cat([hr, x], dim=1)
        hr = torch.nn.functional.gelu(self.decod2(hr))
        hr = torch.cat([hr, x], dim=1)
        hr = torch.nn.functional.gelu(self.decod3(hr))
        hr = torch.nn.functional.gelu(self.decod4(hr))
        p = self.classif(hr)
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")

        return p + 0.2 * plow


class Baseline(torch.nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
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

        self.decod1 = torch.nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.decod2 = torch.nn.Conv2d(224, 128, kernel_size=1)
        self.decod3 = torch.nn.Conv2d(288, 256, kernel_size=1)
        self.decod4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.classif = torch.nn.Conv2d(256, 13, kernel_size=1)

    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.5
        xm = torch.zeros(x.shape[0], 1, 512, 512).cuda()
        x = torch.cat([x, xm], dim=1)

        hr = self.backbone[2](self.backbone[1](self.backbone[0](x)))
        x = self.backbone[5](self.backbone[4](self.backbone[3](hr)))
        plow = self.classiflow(x)
        plow = torch.nn.functional.interpolate(plow, size=(512, 512), mode="bilinear")

        x = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear")
        hr = torch.nn.functional.gelu(self.decod1(hr))
        hr = torch.cat([hr, x], dim=1)
        hr = torch.nn.functional.gelu(self.decod2(hr))
        hr = torch.cat([hr, x], dim=1)
        hr = torch.nn.functional.gelu(self.decod3(hr))
        hr = torch.nn.functional.gelu(self.decod4(hr))
        p = self.classif(hr)
        p = torch.nn.functional.interpolate(p, size=(512, 512), mode="bilinear")

        return p + 0.2 * plow, hr


if __name__ == "__main__":
    net = torch.load("../baselinePP/build/model.pth")

    copynet = Baseline()
    check = set()
    for module in net._modules:
        print(module)
        check.add(module)
        copynet._modules[module] = net._modules[module]

    for module in copynet._modules:
        check.remove(module)
    print("len(check)==", len(check))

    torch.save(copynet, "build/modelB.pth")
