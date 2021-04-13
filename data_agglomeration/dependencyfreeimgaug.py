import torch
import random
import numpy as np


def ensureimage(x, device="cuda"):
    return torch.min(
        torch.ones(x.shape).to(device), torch.max(x, torch.zeros(x.shape).to(device))
    )


def pepperandsalt(x, nbpixel=30, device="cuda"):
    row = np.random.randint(0, x.shape[2], size=nbpixel)
    col = np.random.randint(0, x.shape[3], size=nbpixel)
    pepperorsalt = np.random.randint(0, 2, size=nbpixel)
    for j in range(3):
        for i in range(nbpixel):
            x[:, j, row[i], col[i]] = torch.ones(1).to(device) * float(pepperorsalt[i])
    return x


def shift(x, level=30, device="cuda"):
    for j in range(3):
        x[:, j, :, :] += (torch.rand().to(device) * 2 - 1) * level / 255
    return ensureimage(x)


def blur(x, level=2, device="cuda"):
    return torch.nn.functional.max_pool2d(
        x, kernel_size=2 * level + 1, padding=level, stride=1
    )


def gaussian(x, level=5, device="cuda"):
    return ensureimage(x + torch.randn(x.shape).to(device) * level / 255)


def uniform(x, level=10, device="cuda"):
    return ensureimage(x + (torch.rand(x.shape) * 2 - 1).to(device) * level / 255)


def cutout(x, level=5, device="cuda"):
    row = random.randint(0, x.shape[2] - level - 1)
    col = random.randint(0, x.shape[3] - level - 1)
    x[:, :, row : row + level, col : col + level] = float(random.randint(0, 1))
    return x


def dropchannel(x, device="cuda"):
    x[:, random.randint(0, 2), :, :] = float(random.randint(0, 1))
    return x


def gray(x, device="cuda"):
    tmp = 0.333 * (x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :])
    x[:, 0, :, :] = tmp
    x[:, 1, :, :] = tmp
    x[:, 2, :, :] = tmp
    return x


def augment(x, device="cuda"):
    with torch.no_grad():
        for i in range(x.shape[0] // 2):
            if random.randint(0, 10) == 0:
                # large data augment
                if random.randint(0, 1) == 0:
                    x[i : i + 1] = dropchannel(x[i : i + 1], device=device)
                else:
                    x[i : i + 1] = gray(x[i : i + 1], device=device)
            else:
                # small data augment
                i = random.randint(0, 5)
                if i == 0:
                    x[i : i + 1] = pepperandsalt(x[i : i + 1], device=device)
                if i == 1:
                    x[i : i + 1] = shift(x[i : i + 1], device=device)
                if i == 2:
                    x[i : i + 1] = blur(x[i : i + 1], device=device)
                if i == 3:
                    x[i : i + 1] = gaussian(x[i : i + 1], device=device)
                if i == 4:
                    x[i : i + 1] = uniform(x[i : i + 1], device=device)
                if i == 5:
                    x[i : i + 1] = cutout(x[i : i + 1], device=device)

    x = torch.Tensor(x.cpu().numpy(), requires_grad=true)
    return x
