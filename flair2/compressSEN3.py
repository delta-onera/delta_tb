import torch
import numpy


def bandbasedfilter(x):
    B8a = x[7]*0.9+0.1*x[6]
    B4 = x[2]
    NDVI = (B8a - B4) / (B8a + B4)
    NDWI = (x[1] - B8a) /(x[1] - B8a)
    EVI = 2.5 * ((B8a - B4) / (B8a + 6 * B4 - 7.5 * x[0] + 1))
    SAVI = ((B8a - B4) / (B8a + B4 + 0.5)) * (1.5)
    NBR = (B8a - x[-1]) / (B8a + x[-1])
    UAI = (x[-1] - B4) / (x[-1] + B4)
    NDMI = (B8a - x[-2]) / (B8a + x[-2])
    CI = (B8a - x[6]) / (B8a + x[6])
    BSI = ((x[-2] + B4) - (B8a + x[0])) / (x[-2] + B4 +B8a + x[0])
    NDSI = (x[1] - x[-2]) / (x[1] + x[-2])
    NDCI = (x[0] - x[-2]) / (x[0] + x[-2])
    LAI = 3.618 * ((B8a - B4) / (B8a + B4)) - 0.118
    

def compress(x):
    x = numpy.transpose(x * 1.0, axes=(1, 2, 3, 0))
    x = torch.Tensor(x).cuda()
    B, H, W, T = x.shape
    assert B == 10

    for b in range(B):
        for t in range(T):
            tmp = x[b, :, :, t].flatten()
            tmp, _ = torch.sort(tmp)
            I = int(0.01 * len(tmp))
            vmin, vmax = tmp[I], tmp[-I]
            x[b, :, :, t] = (x[b, :, :, t] - vmin) / (vmax - vmin)

    x = torch.clamp(x, 0, 1).half().float()
    xm, xv = x.mean(3), x.var(3)
    tmp = [((x[:, :, :, t] - xm).abs().sum(), t) for t in range(T)]
    tmp = sorted(tmp)
    xn = x[:, :, :, tmp[0][1]]

    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dxm = dx.mean(3)

    f2 = x[:, :, :, 0 : T // 2].mean(3)
    f3 = x[:, :, :, T // 2 : -1].mean(3)

    f4 = x[:, :, :, 0 : T // 4].mean(3)
    f5 = x[:, :, :, T // 4 : T // 2].mean(3)
    f6 = x[:, :, :, T // 2 : 3 * T // 4].mean(3)
    f7 = x[:, :, :, 3 * T // 4 : -1].mean(3)

    F = torch.cat([xn, xm, xv, dxm, f2, f3, f4, f5, f6, f7], dim=0)
    assert F.shape == (100, H, W)
    return F


root = "/scratchf/CHALLENGE_IGN/FLAIR_2/"
l = ["alltestpaths.pth", "alltrainpaths.pth"]
done = set()
for name in l:
    paths = torch.load(root + name)
    print(len(paths))
    for i in paths:
        if paths[i]["sen"] in done:
            continue
        print(paths[i]["sen"])
        done.add(paths[i]["sen"])
        sentinel = numpy.load(root + paths[i]["sen"])

        if len(done) == 1:
            print(sentinel.shape)
        sentinel = compress(sentinel)
        sentinel = sentinel.cpu().numpy()

        numpy.save(root + paths[i]["sen"], sentinel)

print("GOOOOOOOOOOOD")
