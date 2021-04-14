import torch
import random
import numpy as np


def ensureimage(x, device="cuda"):
    return torch.min(
        torch.ones(x.shape).to(device), torch.max(x, torch.zeros(x.shape).to(device))
    )


def augment(x, device="cuda"):
    with torch.no_grad():
        for i in range(x.shape[0] // 2):
            if random.randint(0, 10) == 0:
                # large data augment
                if random.randint(0, 1) == 0:
                    tmp = random.randint(0, 2)
                    x[i, tmp] *= 0.0
                    x[i, tmp] += float(random.randint(0, 1))
                else:
                    tmp = 0.333 * (x[i, 0, :, :] + x[i, 1, :, :] + x[i, 2, :, :])
                    x[i, 0, :, :] = tmp
                    x[i, 1, :, :] = tmp
                    x[i, 2, :, :] = tmp
            else:
                # small data augment
                j = random.randint(0, 5)
                if j == 0:
                    level = 30
                    row = np.random.randint(0, x.shape[2], size=level)
                    col = np.random.randint(0, x.shape[3], size=level)
                    pepperorsalt = np.random.randint(0, 2, size=level)
                    for j in range(3):
                        for i in range(level):
                            x[i, j, row[i], col[i]] = (
                                torch.ones(1) * float(pepperorsalt[i])
                            ).to(device)
                if j == 1:
                    level = 30.0
                    for j in range(3):
                        tmp = (np.random.rand(1)[0] * 2.0 - 1.0) * level / 255.0
                        x[i, j] += float(tmp)
                if j == 2:
                    x[i : i + 1] = torch.nn.functional.max_pool2d(
                        x[i : i + 1], kernel_size=7, padding=3, stride=1
                    )
                if j == 3:
                    level = 15.0
                    x[i] += (torch.randn(x[i].shape) * level / 255.0).to(device)
                if j == 4:
                    level = 15.0
                    x[i] += ((torch.rand(x[i].shape) * 2.0 - 1.0) * level / 255.0).to(
                        device
                    )
                if j == 5:
                    level = 5
                    row = random.randint(0, x.shape[2] - level - 1)
                    col = random.randint(0, x.shape[3] - level - 1)
                    x[i, :, row : row + level, col : col + level] = float(
                        random.randint(0, 1)
                    )

    x = torch.Tensor(ensureimage(x).cpu().numpy()).to(device)
    x.requires_grad = True
    return x
