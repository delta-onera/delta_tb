import torch
import numpy


def compress(x):
    B2, B3, B4, B5, B6 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    B7, B8, B8a, B11, B12 = x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9]
    B8 = (B8 + B8a) / 2

    NDCI = (B2 - B11) / (B2 + B11)  # cloud
    NDWI = (B3 - B8) / (B3 + B8)  # water
    NDSI = (B3 - B11) / (B3 + B11)  # snow
    UAI = (B12 - B4) / (B12 + B4)  # building

    NDVI = (B8 - B4) / (B8 + B4)  # vegetation
    EVI = 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))  # vegetation again
    BSI = ((B11 + B4) - (B8 + B2)) / (B11 + B4 + B8 + B2)  # vegetation again again
    NDMI = (B8 - B11) / (B8 + B11)  # moisture
    CI = (B8 - B7) / (B8 + B7)  # chlorophyll
    LAI = 3.618 * ((B8 - B4) / (B8 + B4)) - 0.118  # leaf

    f = [NDCI, NDWI, NDSI, UAI, NDVI]
    f = f + [EVI, BSI, NDMI, CI, LAI]
    f = torch.stack(f, dim=0).unsqueeze(0)

    f = torch.nan_to_num(f)
    f = torch.clamp(f, -1, 1)

    _, B, T, H, W = f.shape
    assert B == 10
    if T < 20:
        print(T)
    f = torch.nn.functional.interpolate(f, size=(32, H, W), mode="trilinear")

    f = f[0].half().float()
    f = torch.nan_to_num(f)
    f = torch.clamp(f, -1, 1)
    return f


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
        sentinel = numpy.load(root + paths[i]["sen"]) * 1.0

        if len(done) == 1:
            print(sentinel.shape)
        sentinel = compress(torch.Tensor(sentinel).cuda())
        sentinel = sentinel.cpu().numpy()

        numpy.save(root + paths[i]["sen"], sentinel)

print("GOOOOOOOOOOOD")
