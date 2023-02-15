import torch
import torchvision

with torch.no_grad():
    net1 = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
    tmp = torch.cat([net1[0][0].weight.clone()] * 2, dim=1)

    net2 = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features
    net2[0][0] = torch.nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
    net2[0][0].weight = torch.nn.Parameter(tmp * 0.5)

    a = torch.rand(2, 3, 256, 256)
    b = torch.cat([a] * 2, dim=1)

    f1 = net1[0][0](a)
    f2 = net2[0][0](b)

    print(a.abs().flatten().sum())
    print(f1.abs().flatten().sum())
    print((f2 - f1).abs().flatten().sum())
