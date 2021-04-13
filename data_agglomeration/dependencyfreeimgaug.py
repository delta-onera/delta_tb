import torch
import random
import numpy as np


def ensureimage(x, device="cuda"):
    return torch.min(
        torch.ones(x.shape).to(device), torch.max(x, torch.zeros(x.shape).to(device))
    )


def pepperandsalt(k, x, nbpixel=30, device="cuda"):
    row = np.random.randint(0, x.shape[2], size=nbpixel)
    col = np.random.randint(0, x.shape[3], size=nbpixel)
    pepperorsalt = np.random.randint(0, 2, size=nbpixel)
    for j in range(3):
        for i in range(nbpixel):
            x[k, j, row[i], col[i]] = torch.ones(1).to(device) * float(pepperorsalt[i])


def shift(k, x, level=30, device="cuda"):
    for j in range(3):
        x[k, j, :, :] += (torch.rand().to(device) * 2 - 1) * level / 255


def gaussian(k, x, level=5, device="cuda"):
    x[k] += (torch.randn(x[k].shape) * level / 255).to(device)


def uniform(k, x, level=10, device="cuda"):
    x[k] += ((torch.rand(x[k].shape) * 2 - 1) * level / 255).to(device)


def cutout(k, x, level=5, device="cuda"):
    row = random.randint(0, x.shape[2] - level - 1)
    col = random.randint(0, x.shape[3] - level - 1)
    x[k, :, row : row + level, col : col + level] = float(random.randint(0, 1))


def dropchannel(k, x, device="cuda"):
    x[k, random.randint(0, 2), :, :] = float(random.randint(0, 1))


def gray(k, x, device="cuda"):
    tmp = 0.333 * (x[k, 0, :, :] + x[k, 1, :, :] + x[k, 2, :, :])
    x[k, 0, :, :] = tmp
    x[k, 1, :, :] = tmp
    x[k, 2, :, :] = tmp


def augment(x, device="cuda"):
    with torch.no_grad():
        for i in range(x.shape[0] // 2):
            if random.randint(0, 10) == 0:
                # large data augment
                if random.randint(0, 1) == 0:
                    dropchannel(i, x, device=device)
                else:
                    gray(i, x, device=device)
            else:
                # small data augment
                j = random.randint(0, 5)
                if j == 0:
                    pepperandsalt(i, x, device=device)
                if j == 1:
                    shift(i, x, device=device)
                if j == 2:
                    x[i : i + 1] = torch.nn.functional.max_pool2d(
                        x[i : i + 1], kernel_size=2 * level + 1, padding=level, stride=1
                    )
                if j == 3:
                    gaussian(i, x, device=device)
                if j == 4:
                    uniform(i, x, device=device)
                if j == 5:
                    cutout(i, x, device=device)

    x = torch.Tensor(x.cpu().numpy(), requires_grad=true)
    return x
