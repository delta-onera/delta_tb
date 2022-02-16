import os
import numpy
import PIL
from PIL import Image, ImageDraw
import json
import csv

root = "/scratchf/"
rootminiworld = "/scratchf/miniworld_1M/"

if os.path.exists(rootminiworld):
    os.system("rm -rf " + rootminiworld)
    os.makedirs(rootminiworld)

TARGET_RESOLUTION = 100.0
TODO = {}
TODO["bradbery"] = root + "DATASETS/BRADBURY_BUILDING_HEIGHT/"
TODO["dfc"] = root + "DFC2015/"
TODO["isprs"] = root + "DATASETS/ISPRS_POTSDAM/"
TODO["inria"] = root + "DATASETS/INRIA/AerialImageDataset/train/"
TODO["landcover"] = root + "landcover.ai.v1/"
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
        label = label.resize(size, PIL.Image.NEAREST)
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

        image.save(outpath + "/" + str(i) + "_x.png")
        label.save(outpath + "/" + str(i) + "_y.png")


def resizeallram(outpath, XY, resolution):
    for i, name in enumerate(XY):
        image, label = XY[name]

        image, label = resizenumpy(image=image, label=label, resolution=resolution)

        image.save(outpath + "/" + str(i) + "_x.png")
        label.save(outpath + "/" + str(i) + "_y.png")


def resize_BRADBURY_BUILDING_HEIGHT(outpath, inpath, resolution):
    if "Atlanta_01_buildingCoord" in inpath[1]:
        inpath[1] = inpath[1].replace(
            "Atlanta_01_buildingCoord", "Atlanta_01buildingCoord"
        )

    image = PIL.Image.open(inpath[0]).convert("RGB").copy()
    label = Image.new("RGB", image.size)

    draw = ImageDraw.Draw(label)
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

                resize_BRADBURY_BUILDING_HEIGHT(out, inpath, resolution[town] * 100)


if "dfc" in TODO:
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


if "isprs" in TODO:
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


if "inria" in TODO:
    print("export inria")
    towns = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
    for town in towns:
        makepath(town)

        XY = {}
        for i in range(1, 21):
            XY[i] = ["images/" + town + str(i), "gt/" + town + str(i)]
            XY[i] = [XY[i][0] + ".tif", XY[i][1] + ".tif"]
        tmp = rootminiworld + town + "/train/"
        resizeall(tmp, TODO["inria"], XY, 30)

        XY = {}
        for i in range(21, 21 + 15):
            XY[i] = ["images/" + town + str(i), "gt/" + town + str(i)]
            XY[i] = [XY[i][0] + ".tif", XY[i][1] + ".tif"]
        tmp = rootminiworld + town + "/test/"
        resizeall(tmp, TODO["inria"], XY, 30)

if "landcover" in TODO:
    print("export landcover")
    makepath("pologne")

    allname = os.listdir(TODO["landcover"] + "/masks")
    allname = [name for name in allname if ".tif" in name]

    half = [
        "N-33-119-C-c-3-3.tif",
        "N-33-60-D-c-4-2.tif",
        "N-33-60-D-d-1-2.tif",
        "N-33-96-D-d-1-1.tif",
        "N-34-61-B-a-1-1.tif",
        "N-34-66-C-c-4-3.tif",
        "N-34-97-C-b-1-2.tif",
        "N-34-97-D-c-2-4.tif",
    ]
    half = sorted(half)
    allname = sorted([name for name in allname if name not in half])

    split = int(len(allname) * 0.66)
    names = {}
    names["train"] = allname[0:split] + half[0:4]
    names["test"] = allname[split:] + half[4:]

    for flag in ["train", "test"]:
        for i, name in enumerate(names[flag]):
            x = PIL.Image.open(TODO["landcover"] + "images/" + name)
            y = PIL.Image.open(TODO["landcover"] + "masks/" + name)

            x = numpy.uint8(numpy.asarray(x.convert("RGB").copy()))
            y = numpy.asarray(y.convert("L").copy())
            y = numpy.uint8(y == 1) * 255

            if name in half:
                x, y = resizenumpy(image=x, label=y, resolution=50)
            else:
                x, y = resizenumpy(image=x, label=y, resolution=25)

            tmp = rootminiworld + "pologne/" + flag
            x.save(tmp + "/" + str(i) + "_x.png")
            y.save(tmp + "/" + str(i) + "_y.png")

if "airs" in TODO:
    print("export airs")
    makepath("christchurch")

    for flag, flag2 in [("test", "val"), ("train", "train")]:
        XY = {}
        allname = os.listdir(TODO["airs"] + flag2 + "/image")
        for name in allname:
            XY[name] = ["image/" + name[0:-4], "label/" + name[0:-4] + "_vis"]
            XY[name] = [XY[name][0] + ".tif", XY[name][1] + ".tif"]
        tmp = rootminiworld + "christchurch/" + flag
        resizeall(tmp, TODO["airs"] + flag2, XY, 7.5)


import rasterio


def resize_spacenet1(outpath, inpath, XY):
    for i, name in enumerate(XY):
        x, y = XY[name]

        with rasterio.open(inpath + x) as src:
            affine = src.transform
            r = numpy.int16(src.read(1))

        label = Image.new("RGB", (r.shape[1], r.shape[0]))
        draw = ImageDraw.Draw(label)

        with open(inpath + y, "r") as infile:
            text = json.load(infile)
        shapes = text["features"]
        for shape in shapes:
            polygonXYZ = shape["geometry"]["coordinates"][0]
            if type(polygonXYZ) != type([]):
                continue
            if len(polygonXYZ) < 3:
                continue
            polygon = []
            for xyz in polygonXYZ:
                polygon.append(rasterio.transform.rowcol(affine, xyz[0], xyz[1]))
            polygon = [(y, x) for x, y in polygon]
            draw.polygon(polygon, fill="#ffffff", outline="#ffffff")

        image = PIL.Image.open(inpath + "/" + x).convert("RGB").copy()

        image, label = resize(image=image, label=label)
        label.save(outpath + "/" + str(i) + "_y.png")
        image.save(outpath + "/" + str(i) + "_x.png")


if "spacenet1" in TODO:
    print("export spacenet1")
    makepath("rio")

    allname = os.listdir(TODO["spacenet1"] + "3band")
    allname = [name for name in allname if name[-4 : len(name)] == ".tif"]
    allname = sorted([name[5:-4] for name in allname])
    split = int(len(allname) * 0.66)
    names = {}
    names["train"] = allname[0:split]
    names["test"] = allname[split:]

    for flag in ["train", "test"]:
        XY = {}
        for name in names[flag]:
            XY[name] = [" ", " "]
            XY[name][0] = "3band/3band" + name + ".tif"
            XY[name][1] = "geojson/Geo" + name + ".geojson"

        tmp = rootminiworld + "rio/" + flag + "/"
        resize_spacenet1(tmp, TODO["spacenet1"], XY)
