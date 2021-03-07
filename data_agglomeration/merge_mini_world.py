import os
import numpy as np
import PIL
from PIL import Image


def resizefile(root, XY, output, nativeresolution, outputresolution=50.0):
    i = 0
    for name in XY:
        x, y = XY[name]
        image = PIL.Image.open(root + "/" + x).convert("RGB").copy()
        label = PIL.Image.open(root + "/" + y).convert("L").copy()

        if nativeresolution != outputresolution:
            image = image.resize(
                (
                    int(image.size[0] * nativeresolution / outputresolution),
                    int(image.size[1] * nativeresolution / outputresolution),
                ),
                PIL.Image.BILINEAR,
            )
            label = label.resize((image.size[0], image.size[1]), PIL.Image.NEAREST)

        image.save(output + "/" + str(i) + "_x.png")
        label.save(output + "/" + str(i) + "_y.png")
        i += 1


def resizeram(XY, output, nativeresolution, outputresolution=50):
    i = 0
    for name in XY:
        x, y = XY[name]
        image = PIL.Image.fromarray(np.uint8(x))
        label = PIL.Image.fromarray(np.uint8(y))

        if nativeresolution != outputresolution:
            image = image.resize(
                (
                    int(image.size[0] * nativeresolution / outputresolution),
                    int(image.size[1] * nativeresolution / outputresolution),
                ),
                PIL.Image.BILINEAR,
            )
            label = label.resize((image.size[0], image.size[1]), PIL.Image.NEAREST)

        image.save(output + "/" + str(i) + "_x.png")
        label.save(output + "/" + str(i) + "_y.png")
        i += 1


whereIam = os.uname()[1]
if whereIam == "super":
    availabledata = ["semcity", "isprs", "dfc"]
    root = "/data/"

if whereIam == "wdtim719z":
    availabledata = ["semcity", "isprs", "airs", "dfc"]
    root = "/data/"

if whereIam == "ldtis706z":
    availabledata = ["semcity", "isprs", "airs", "dfc", "inria"]
    root = "/media/achanhon/bigdata/data/"

if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    availabledata = ["semcity", "isprs", "airs", "dfc", "inria", "mini"]
    root = "TODO"


def makepath(name):
    os.makedirs(root + "miniworld/" + name)
    os.makedirs(root + "miniworld/" + name + "/train")
    os.makedirs(root + "miniworld/" + name + "/test")


if "inria" in availabledata:
    print("export inria")
    towns = ["austin", "chicago", "kitsap", "tyrol", "vienna"]
    for town in towns:
        makepath(town)

        XY = {}
        for i in range(20):
            XY[i] = (
                "images/" + town + str(1 + i) + ".tif",
                "gt/" + town + str(1 + i) + ".tif",
            )
        resizefile(
            root + "INRIA/AerialImageDataset/train/",
            XY,
            root + "miniworld/" + town + "/train/",
            30,
        )

        XY = {}
        for i in range(15):
            XY[i] = (
                "images/" + town + str(21 + i) + ".tif",
                "gt/" + town + str(21 + i) + ".tif",
            )
        resizefile(
            root + "INRIA/AerialImageDataset/train/",
            XY,
            root + "miniworld/" + town + "/test/",
            30,
        )

if "dfc" in availabledata:
    print("export dfc 2015 bruges")
    makepath("bruges")

    names = {}
    names["train"] = ["315130_56865", "315130_56870", "315135_56870", "315140_56865"]
    names["test"] = ["315135_56865", "315145_56865"]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            x = (
                PIL.Image.open(
                    root + "DFC2015/" + "BE_ORTHO_27032011_" + name + ".tif"
                )
                .convert("RGB")
                .copy()
            )
            y = (
                PIL.Image.open(root + "DFC2015/" + "label_" + name + ".tif")
                .convert("RGB")
                .copy()
            )
            y = np.uint8(np.asarray(y))
            y = (
                np.uint8(y[:, :, 0] == 0)
                * np.uint8(y[:, :, 1] == 0)
                * np.uint8(y[:, :, 2] == 255)
                * 255
            )

            XY[name] = (x, y)

        resizeram(XY, root + "miniworld/bruges/" + flag, 5)

if "airs" in availabledata:
    print("export airs")
    makepath("christchurch")

    for flag, flag2 in [("test", "val"), ("train", "train")]:
        XY = {}
        allname = os.listdir(root + "AIRS/" + flag2 + "/image")
        for name in allname:
            XY[name] = (
                "image/" + name[0:-4] + ".tif",
                "label/" + name[0:-4] + "_vis.tif",
            )
        resizefile(
            root + "AIRS/" + flag2,
            XY,
            root + "miniworld/christchurch/" + flag + "/",
            7.5,
        )

if "isprs" in availabledata:
    print("export isprs potsdam")
    makepath("potsdam")

    names = {}
    names["train"] = [
        "top_potsdam_2_10_",
        "top_potsdam_2_11_",
        "top_potsdam_2_12_",
        "top_potsdam_3_10_",
        "top_potsdam_3_11_",
        "top_potsdam_3_12_",
        "top_potsdam_4_10_",
        "top_potsdam_4_11_",
        "top_potsdam_4_12_",
        "top_potsdam_5_10_",
        "top_potsdam_5_11_",
        "top_potsdam_5_12_",
        "top_potsdam_6_7_",
        "top_potsdam_6_8_",
    ]
    names["test"] = [
        "top_potsdam_6_9_",
        "top_potsdam_6_10_",
        "top_potsdam_6_11_",
        "top_potsdam_6_12_",
        "top_potsdam_7_7_",
        "top_potsdam_7_8_",
        "top_potsdam_7_9_",
        "top_potsdam_7_10_",
        "top_potsdam_7_11_",
        "top_potsdam_7_12_",
    ]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            x = (
                PIL.Image.open(
                    root + "ISPRS_POTSDAM/" + "2_Ortho_RGB/" + name + "RGB.tif"
                )
                .convert("RGB")
                .copy()
            )
            y = (
                PIL.Image.open(
                    root
                    + "ISPRS_POTSDAM/"
                    + "5_Labels_for_participants/"
                    + name
                    + "label.tif"
                )
                .convert("RGB")
                .copy()
            )
            y = np.uint8(np.asarray(y))
            y = (
                np.uint8(y[:, :, 0] == 0)
                * np.uint8(y[:, :, 1] == 0)
                * np.uint8(y[:, :, 2] == 255)
                * 255
            )

            XY[name] = (x, y)

        resizeram(XY, root + "miniworld/potsdam/" + flag, 5)


import rasterio


def histogramnormalization(im, removecentiles=2, tile=0, stride=0, vmin=1, vmax=-1):
    print("extracting pivot")
    if tile <= 0 or stride <= 0 or tile > stride:
        allvalues = list(im.flatten())
    else:
        allvalues = []
        for row in range(0, im.shape[0] - tile, stride):
            for col in range(0, im.shape[1] - tile, stride):
                allvalues += list(im[row : row + tile, col : col + tile].flatten())

    ## remove "no data"
    if vmin < vmax:
        allvalues = [v for v in allvalues if vmin <= v and v <= vmax]

    print("sorting pivot")
    allvalues = sorted(allvalues)
    n = len(allvalues)
    allvalues = allvalues[0 : int((100 - removecentiles) * n / 100)]
    allvalues = allvalues[int(removecentiles * n / 100) :]

    n = len(allvalues)
    k = n // 255
    pivot = [0] + [allvalues[i] for i in range(0, n, k)]
    assert len(pivot) >= 255

    print("normalization")
    out = np.uint8(np.zeros(im.shape, dtype=int))
    for i in range(1, 255):
        if i % 10 == 0:
            print("normalization in progress", i, "/255")
        out = np.maximum(out, np.uint8(im > pivot[i]) * i)

    print("normalization succeed")
    return np.uint8(out)


if "semcity" in availabledata:
    print("export toulouse")
    makepath("toulouse")

    names = {}
    names["train"] = ["04", "08"]
    names["test"] = ["03", "07"]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:

            ###physical image which requires normalization
            src = rasterio.open(root + "SEMCITY_TOULOUSE/TLS_BDSD_M_" + name + ".tif")
            r = histogramnormalization(np.int16(src.read(4)))
            g = histogramnormalization(np.int16(src.read(3)))
            b = histogramnormalization(np.int16(src.read(2)))

            x = np.stack([r, g, b], axis=2)

            y = (
                PIL.Image.open(root + "SEMCITY_TOULOUSE/TLS_GT_" + name + ".tif")
                .convert("RGB")
                .copy()
            )
            y = np.uint8(np.asarray(y))
            y = (
                np.uint8(y[:, :, 0] == 238)
                * np.uint8(y[:, :, 1] == 118)
                * np.uint8(y[:, :, 2] == 33)
                * 255
            )

            XY[name] = (x, y)

        resizeram(XY, root + "miniworld/toulouse/" + flag, 50)
