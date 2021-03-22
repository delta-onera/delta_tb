import os
import numpy as np
import json
import PIL
from PIL import Image
import rasterio


# for not uint8 image !
def histogramnormalization(
    im, removecentiles=2, tile=0, stride=0, vmin=1, vmax=-1, verbose=True
):
    if verbose:
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

    if verbose:
        print("sorting pivot")
    allvalues = sorted(allvalues)
    n = len(allvalues)
    allvalues = allvalues[0 : int((100 - removecentiles) * n / 100)]
    allvalues = allvalues[int(removecentiles * n / 100) :]

    n = len(allvalues)
    k = n // 255
    pivot = [0] + [allvalues[i] for i in range(0, n, k)]
    assert len(pivot) >= 255

    if verbose:
        print("normalization")
    out = np.uint8(np.zeros(im.shape, dtype=int))
    for i in range(1, 255):
        if i % 10 == 0 and verbose:
            print("normalization in progress", i, "/255")
        out = np.maximum(out, np.uint8(im > pivot[i]) * i)

    if verbose:
        print("normalization succeed")
    return np.uint8(out)


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

        tmp = np.asarray(label)
        if np.sum(tmp) != 0:
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


def scratchfilespacenet2(root, XY, output):
    i = 0
    for name in XY:
        x, y = XY[name]

        with open(root+y, "r") as infile:
            text = json.load(infile)
        shapes = text["features"]

        with rasterio.open(root+x) as src:
            affine = src.transform
            r = histogramnormalization(np.int16(src.read(1)), verbose=False)
            g = histogramnormalization(np.int16(src.read(2)), verbose=False)
            b = histogramnormalization(np.int16(src.read(3)), verbose=False)

        mask = Image.new("RGB", (r.shape[0], r.shape[1]))

        draw = ImageDraw.Draw(mask)
        for shape in shapes:
            polygonXYZ = shape["geometry"]["coordinates"][0]
            polygon = [
                rasterio.transform.rowcol(affine, xyz[0], xyz[1]) for xyz in polygonXYZ
            ]
            polygon = [(y, x) for x, y in polygon]
            draw.polygon(polygon, fill="#ffffff", outline="#ffffff")

        mask.save(output + str(i) + "_y.png")

        x = np.stack([r, g, b], axis=2)
        image = Image.fromarray(x)
        image.save(output + str(i) + "_x.png")

        i += 1


whereIam = os.uname()[1]
if whereIam == "super":
    availabledata = ["isprs", "semcity"]
    root = "/data/"
    rootminiworld = root + "/miniworld/"

if whereIam == "wdtim719z":
    availabledata = ["semcity", "isprs", "airs", "dfc"]
    root = "/data/"
    rootminiworld = root + "/miniworld/"

if whereIam == "ldtis706z":
    availabledata = ["semcity", "isprs", "airs", "dfc", "inria"]
    root = "/media/achanhon/bigdata/data/"
    rootminiworld = root + "/miniworld/"

if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    availabledata = ["spacenet2", "isprs", "airs", "inria"]
    root = "/scratch_ai4geo/DATASETS/"
    rootminiworld = "/scratch_ai4geo/miniworldtmp/"


def makepath(name):
    os.makedirs(rootminiworld + name)
    os.makedirs(rootminiworld + name + "/train")
    os.makedirs(rootminiworld + name + "/test")


if "spacenet2" in availabledata:
    print("export spacenet2")
    towns = [
        ("2_Vegas", "vegas"),
        ("3_Paris", "paris"),
        ("4_Shanghai", "shanghai"),
        ("5_Khartoum", "khartoum"),
    ]
    for town, out in towns:
        makepath(out)

        allname = os.listdir(
            root + "SPACENET2/train/AOI_" + town + "_Train/RGB-PanSharpen"
        )
        allname = sorted([name[0:-4] for name in allname])
        split = int(len(allname) * 0.66)
        names = {}
        names["train"] = allname[0:split]
        names["test"] = allname[split : len(allname)]

        for flag in ["train", "test"]:
            XY = {}
            for name in names[flag]:
                XY[name] = (
                    "AOI_2_" + town + "_Train/RGB-PanSharpen/" + name + ".tif",
                    "AOI_2_" + town + "_Train/geojson/building/" + name + ".geojson",
                )
            scratchfilespacenet2(
                root + "SPACENET2/train/",
                XY,
                rootminiworld + out + "/" + flag + "/",
            )

if "inria" in availabledata:
    print("export inria")
    towns = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
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
            rootminiworld + town + "/train/",
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
            rootminiworld + town + "/test/",
            30,
        )

if "airs" in availabledata:
    print("export airs")
    makepath("christchurch")

    hack = ""
    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        hack = "trainval/"

    for flag, flag2 in [("test", "val"), ("train", "train")]:
        XY = {}
        allname = os.listdir(root + "AIRS/" + hack + flag2 + "/image")
        for name in allname:
            XY[name] = (
                "image/" + name[0:-4] + ".tif",
                "label/" + name[0:-4] + "_vis.tif",
            )
        resizefile(
            root + "AIRS/" + hack + flag2,
            XY,
            rootminiworld + "christchurch/" + flag + "/",
            7.5,
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
                PIL.Image.open(root + "DFC2015/" + "BE_ORTHO_27032011_" + name + ".tif")
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

        resizeram(XY, rootminiworld + +"bruges/" + flag, 5)

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

        resizeram(XY, rootminiworld + "potsdam/" + flag, 5)


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

        resizeram(XY, rootminiworld + "toulouse/" + flag, 50)
