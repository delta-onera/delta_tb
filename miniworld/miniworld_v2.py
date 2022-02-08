import os
import numpy
import PIL
from PIL import Image, ImageDraw
import json
import csv

root = "/scratchf/"
rootminiworld = "/scratchf/miniworldtmp/"
TARGET_RESOLUTION = 50.0
TODO = {}
TODO["bradbery"] = root + "DATASETS/BRADBURY_BUILDING_HEIGHT/"
TODO["dfc"] = root + "DATASETS/DFC2015/"
TODO["isprs"] = root + "DATASETS/ISPRS_POTSDAM/"
TODO["inria"] = root + "DATASETS/INRIA/"
TODO["airs"] = root + "DATASETS/AIRS/trainval/"
TODO["spacenet1"] = root + "DATASETS/SPACENET1/train/"


def makepath(name):
    os.makedirs(rootminiworld + name)
    os.makedirs(rootminiworld + name + "/train")
    os.makedirs(rootminiworld + name + "/test")


def resize(image=None, label=None, resolution=50.0):
    if resolution == TARGET_RESOLUTION:
        return image, label
    coef = resolution / TARGET_RESOLUTION
    size = (int(image.size[0] * coef), int(image.size[1] * coef))

    if image is not None:
        image = image.resize(size, PIL.Image.BILINEAR)
    if label is not None:
        label = image.resize(size, PIL.Image.NEAREST)
    return image, label


def resizenumpy(image=None, label=None, resolution=50.0):
    if image is not None:
        image = PIL.Image.fromarray(numpy.uint8(image))
    if label is not None:
        label = PIL.Image.fromarray(numpy.uint8(label))

    image, label = resize(image=image, label=label, resolution=resolution)
    return image, label


def resizeall(outpath, inpath, XY, resolution):
    for i, name in enumerate(XY):
        x, y = XY[name]
        image = PIL.Image.open(inpath + "/" + x).convert("RGB").copy()
        label = PIL.Image.open(inpath + "/" + y).convert("L").copy()

        image, label = resize(image=image, label=label, resolution=resolution)

        image.save(output + "/" + str(i) + "_x.png")
        label.save(output + "/" + str(i) + "_y.png")


def resizeallram(outpath, XY, resolution):
    for i, name in enumerate(XY):
        image, label = XY[name]

        image, label = resizenumpy(image=image, label=label, resolution=resolution)

        image.save(output + "/" + str(i) + "_x.png")
        label.save(output + "/" + str(i) + "_y.png")


def resize_BRADBURY_BUILDING_HEIGHT(outpath, inpath, resolution):
    if "Atlanta_01_buildingCoord" in inpath[1]:
        inpath[1] = inpath[1].replace(
            "Atlanta_01_buildingCoord", "Atlanta_01buildingCoord"
        )

    image = PIL.Image.open(inpath[0]).convert("RGB").copy()
    mask = Image.new("RGB", image.size)

    draw = ImageDraw.Draw(mask)
    with open(inpath[1], newline="") as csvfile:
        csvlines = csv.reader(csvfile, delimiter=",")
        for line in csvlines:
            if line[0] == "Image_Name":
                continue

            polygon = []
            i = 3
            while line[i] != "" and line[i] != "NaN":
                x = int(float(line[i]))
                i += 1
                y = int(float(line[i]))
                i += 1
                polygon.append((x, y))

            draw.polygon(polygon, fill="#ffffff", outline="#ffffff")

    image, label = resize(image=image, label=label, resolution=resolution)
    image.save(outpath[0])
    label.save(outpath[1])


if "bradbery" in TODO:
    print("export bradbery")
    resolution = {}
    resolution["Arlington"] = 0.3
    resolution["Austin"] = 0.5 * 0.3
    resolution["DC"] = 0.16
    resolution["Atlanta"] = 0.5 * 0.3
    resolution["NewHaven"] = 0.3
    resolution["NewYork"] = 0.5 * 0.3
    resolution["Norfolk"] = 0.3
    resolution["SanFrancisco"] = 0.3
    resolution["Seekonk"] = 0.3

    for town in resolution:
        makepath(town)
        tmpIN = TODO["bradbery"] + town + "/" + town + "_0"

        split = {}
        split["train"] = [1, 2]
        split["test"] = [3]
        if town in ["DC", "NewHaven"]:
            split["train"] = [1]
            split["test"] = [2]

        for flag in ["train", "test"]:
            for i in split[flag]:
                imagepath = tmpIN + str(i) + ".tif"
                csvpath = tmpIN + str(i) + "_buildingCoord.csv"

                inpath = [imagepath, csvpath]

                tmp = rootminiworld + town + "/" + flag + "/"
                if flag == "test":
                    out = [tmp + "0_x.png", tmp + "0_y.png"]
                else:
                    out = [tmp + str(i - 1) + "_x.png", tmp + str(i - 1) + "_y.png"]

                resize_BRADBURY_BUILDING_HEIGHT(out, inpath, resolution[town])


if "dfc" in availabledata:
    print("export dfc 2015 bruges")
    makepath("bruges")

    names = {}
    names["train"] = ["315130_56865", "315130_56870", "315135_56870", "315140_56865"]
    names["test"] = ["315135_56865", "315145_56865"]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            x = PIL.Image.open(TODO["dfc"] + "BE_ORTHO_27032011_" + name + ".tif")
            y = PIL.Image.open(TODO["dfc"] + "label_" + name + ".tif")

            x = numpy.uint8(numpy.asarray(x.convert("RGB").copy()))
            y = numpy.asarray(y.convert("RGB").copy())
            y = (y[:, :, 0] == 0) * (y[:, :, 1] == 0) * (y[:, :, 2] == 255) * 255
            XY[name] = (x, numpy.uint8(y))

        resizeallram(rootminiworld + "bruges/" + flag, XY, 5)


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
    names["path"] = "5_Labels_for_participants/"

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            x = PIL.Image.open(TODO["isprs"] + "2_Ortho_RGB/" + name + "RGB.tif")
            y = PIL.Image.open(TODO["isprs"] + names["path"] + name + "label.tif")

            x = numpy.uint8(numpy.asarray(x.convert("RGB").copy()))
            y = numpy.asarray(y.convert("RGB").copy())
            y = (y[:, :, 0] == 0) * (y[:, :, 1] == 0) * (y[:, :, 2] == 255) * 255
            XY[name] = (x, numpy.uint8(y))

        resizeallram(rootminiworld + "potsdam/" + flag, XY, 5)


if "inria" in availabledata:
    print("export inria")
    towns = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
    for town in towns:
        makepath(town)

        XY = {}
        for i in range(1, 21):
            XY[i] = ["images/" + town + str(i), "gt/" + town + str(i)]
            XY[i] = [XY[i][0] + ".tif", XY[i][1] + ".tif"]
        tmp = rootminiworld + town + "/train/"
        resizeall(tmp, TODO[inria], XY, 30)

        XY = {}
        for i in range(21, 21 + 15):
            XY[i] = ["images/" + town + str(i), "gt/" + town + str(i)]
            XY[i] = [XY[i][0] + ".tif", XY[i][1] + ".tif"]
        tmp = rootminiworld + town + "/test/"
        resizeall(tmp, TODO[inria], XY, 30)

if "airs" in availabledata:
    print("export airs")
    makepath("christchurch")

    for flag, flag2 in [("test", "val"), ("train", "train")]:
        XY = {}
        allname = os.listdir(TODO["AIRS"] + flag2 + "/image")
        for name in allname:
            XY[name] = ["image/" + name[0:-4], "label/" + name[0:-4] + "_vis"]
            XY[name] = [XY[i][0] + ".tif", XY[i][1] + ".tif"]
        tmp = rootminiworld + "christchurch/" + flag
        resizeall(tmp, TODO["AIRS"] + flag2, XY, 7.5)


import rasterio


def scratchfilespacenet1(root, XY, output):
    i = 0
    for name in XY:
        x, y = XY[name]

        with open(root + y, "r") as infile:
            text = json.load(infile)
        shapes = text["features"]

        with rasterio.open(root + x) as src:
            affine = src.transform
            r = numpy.int16(src.read(1))

        mask = Image.new("RGB", (r.shape[1], r.shape[0]))

        draw = ImageDraw.Draw(mask)
        for shape in shapes:
            polygonXYZ = shape["geometry"]["coordinates"][0]
            if type(polygonXYZ) != type([]):
                continue
            if len(polygonXYZ) < 3:
                continue
            polygon = [
                rasterio.transform.rowcol(affine, xyz[0], xyz[1]) for xyz in polygonXYZ
            ]
            polygon = [(y, x) for x, y in polygon]
            draw.polygon(polygon, fill="#ffffff", outline="#ffffff")

        mask.save(output + str(i) + "_y.png")

        image = PIL.Image.open(root + "/" + x).convert("RGB").copy()
        image.save(output + "/" + str(i) + "_x.png")

        i += 1


if "spacenet1" in availabledata:
    print("export spacenet1")
    makepath("rio")

    allname = os.listdir(root + "SPACENET1/train/3band")
    allname = [name for name in allname if name[-4 : len(name)] == ".tif"]
    allname = sorted([name[5:-4] for name in allname])
    split = int(len(allname) * 0.66)
    names = {}
    names["train"] = allname[0:split]
    names["test"] = allname[split : len(allname)]

    print("start file processing")
    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            XY[name] = (
                "3band/3band" + name + ".tif",
                "geojson/Geo" + name + ".geojson",
            )
        scratchfilespacenet1(rootminiworld + "rio/", TODO["spacenet1"], XY, +flag + "/")


def scratchfilespacenet2(root, XY, output, pivots):
    i = 0
    for name in XY:
        x, y = XY[name]

        with open(root + y, "r") as infile:
            text = json.load(infile)
        shapes = text["features"]

        with rasterio.open(root + x) as src:
            affine = src.transform
            r = histogramnormalization(
                numpy.int16(src.read(1)), verbose=False, pivot=pivots["r"]
            )
            g = histogramnormalization(
                numpy.int16(src.read(2)), verbose=False, pivot=pivots["g"]
            )
            b = histogramnormalization(
                numpy.int16(src.read(3)), verbose=False, pivot=pivots["b"]
            )

        mask = Image.new("RGB", (r.shape[1], r.shape[0]))

        draw = ImageDraw.Draw(mask)
        for shape in shapes:
            polygonXYZ = shape["geometry"]["coordinates"][0]
            if type(polygonXYZ) != type([]):
                continue
            if len(polygonXYZ) < 3:
                continue
            polygon = [
                rasterio.transform.rowcol(affine, xyz[0], xyz[1]) for xyz in polygonXYZ
            ]
            polygon = [(y, x) for x, y in polygon]
            draw.polygon(polygon, fill="#ffffff", outline="#ffffff")

        mask = mask.resize(
            (
                int(mask.size[0] * 30.0 / 50),
                int(mask.size[1] * 30.0 / 50),
            ),
            PIL.Image.NEAREST,
        )
        mask.save(output + str(i) + "_y.png")

        x = numpy.stack([r, g, b], axis=2)
        image = Image.fromarray(x)
        image = image.resize((mask.size[0], mask.size[1]), PIL.Image.BILINEAR)

        image.save(output + str(i) + "_x.png")

        i += 1


if "semcity" in availabledata:
    print("export toulouse")
    makepath("toulouse")

    hack = ""
    if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
        hack = "../"

    names = {}
    names["train"] = ["04", "08"]
    names["test"] = ["03", "07"]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:

            ###physical image which requires normalization
            src = rasterio.open(
                root + hack + "SEMCITY_TOULOUSE/TLS_BDSD_M_" + name + ".tif"
            )
            r = histogramnormalization(numpy.int16(src.read(4)))
            g = histogramnormalization(numpy.int16(src.read(3)))
            b = histogramnormalization(numpy.int16(src.read(2)))

            x = numpy.stack([r, g, b], axis=2)

            y = (
                PIL.Image.open(root + hack + "SEMCITY_TOULOUSE/TLS_GT_" + name + ".tif")
                .convert("RGB")
                .copy()
            )
            y = numpy.uint8(numpy.asarray(y))
            y = (
                numpy.uint8(y[:, :, 0] == 238)
                * numpy.uint8(y[:, :, 1] == 118)
                * numpy.uint8(y[:, :, 2] == 33)
                * 255
            )

            XY[name] = (x, y)

        resizeram(XY, rootminiworld + "toulouse/" + flag, 50)

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
        allname = [name for name in allname if name[-4 : len(name)] == ".tif"]
        allname = sorted([name[14:-4] for name in allname])
        split = int(len(allname) * 0.66)
        names = {}
        names["train"] = allname[0:split]

        print("collect stats for normalization")
        pivots = {}
        for c in ["r", "g", "b"]:
            pivots[c] = []

        for i in range(0, len(names["train"]), 4):
            with rasterio.open(
                root
                + "SPACENET2/train/AOI_"
                + town
                + "_Train/RGB-PanSharpen/RGB-PanSharpen"
                + names["train"][i]
                + ".tif"
            ) as src:
                r = numpy.int16(src.read(1))
                g = numpy.int16(src.read(2))
                b = numpy.int16(src.read(3))
                pivots["r"] += list(r.flatten())
                pivots["g"] += list(g.flatten())
                pivots["b"] += list(b.flatten())

        print("compute global pivots for normalization")
        for c in ["r", "g", "b"]:
            pivots[c] = [v for v in pivots[c] if v >= 2]
            pivots[c] = sorted(pivots[c])
            n = len(pivots[c])
            pivots[c] = pivots[c][0 : int((100 - 4) * n / 100)]
            pivots[c] = pivots[c][int(4 * n / 100) :]

            n = len(pivots[c])
            k = n // 255

            pivots[c] = [0] + [pivots[c][i] for i in range(0, n, k)]

            assert len(pivots[c]) >= 255

        names["test"] = allname[split : len(allname)]

        print("start file processing")
        for flag in ["train", "test"]:
            XY = {}
            for name in names[flag]:
                XY[name] = (
                    "AOI_"
                    + town
                    + "_Train/RGB-PanSharpen/RGB-PanSharpen"
                    + name
                    + ".tif",
                    "AOI_"
                    + town
                    + "_Train/geojson/buildings/buildings"
                    + name
                    + ".geojson",
                )
            scratchfilespacenet2(
                root + "SPACENET2/train/",
                XY,
                rootminiworld + out + "/" + flag + "/",
                pivots,
            )
