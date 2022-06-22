import numpy
import torch


class Sobel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        G = torch.stack([Gx, Gy])
        self.G = G.unsqueeze(1).float()

    def forward(self, yz):
        tmp = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        tmp.weight = torch.nn.Parameter(self.G.clone(), requires_grad=True)
        tmp.cuda()

        x = tmp(yz)
        norm = torch.sqrt(torch.sum(x * x, dim=1))
        norm = torch.stack([norm, norm], dim=1)
        x = x / (norm + 0.001)
        return x, (norm[:, 0].detach().clone() > 0.0001).int()


def getborder(y):
    sobel = Sobel()
    _, yy = sobel(y.unsqueeze(0).unsqueeze(0))
    return yy[0]


def perf(cm):
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return torch.Tensor((iou0 + iou1, accu, iou0 * 2, iou1 * 2))


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def torchTOpil(x):
    return numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))
