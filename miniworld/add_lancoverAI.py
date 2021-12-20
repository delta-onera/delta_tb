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

        tmp = np.asarray(label)

        tmp = np.uint8(tmp == 1) * 255
        label = PIL.Image.fromarray(tmp)

        if np.sum(tmp) != 0:
            image.save(output + "/" + str(i) + "_x.png")
            label.save(output + "/" + str(i) + "_y.png")
            i += 1


whereIam = os.uname()[1]

path = "/home/achanhon/Téléchargements/landcover.ai.v1"
if whereIam == "super":
    rootminiworld = "build/miniworld/"

if whereIam == "wdtim719z":
    rootminiworld = "build/miniworld/"

if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    rootminiworld = "build/miniworld/"


def makepath(name):
    os.makedirs(rootminiworld + name)
    os.makedirs(rootminiworld + name + "/train")
    os.makedirs(rootminiworld + name + "/test")


organisefiles = {}
organisefiles["Preczow"] = [
    "M-34-65-D-a-4-4",
    "M-34-65-D-c-4-2",
    "M-34-65-D-d-4-1",
    "M-34-77-B-c-2-3",
    "M-34-51-C-b-2-1",
    "M-34-51-C-d-4-1",
]  # M-34-51-C-d-4-1.tif
organisefiles["Grzedy"] = [
    "N-34-94-A-b-2-4",
    "N-34-106-A-b-3-4",
    "N-34-106-A-c-1-3",
]  # N-34-94-A-b-2-4.tif
organisefiles["Fordon"] = [
    "N-34-97-C-b-1-2",
    "N-34-97-D-c-2-4",
    "N-33-96-D-d-1-1",
    "N-33-60-D-c-4-2",
    "N-34-61-B-a-1-1",
    "N-33-60-D-d-1-2",
]  # N-34-97-C-b-1-2.tif
organisefiles["Predocin"] = [
    "M-33-20-D-c-4-2",
    "M-33-20-D-d-3-3",
    "M-33-32-B-b-4-4",
    "N-33-139-C-d-2-2",
    "N-33-139-C-d-2-4",
    "N-33-139-D-c-1-3",
    "M-33-7-A-d-2-3",
    "M-33-7-A-d-3-2",
    "M-33-48-A-c-4-4",
]  # M-33-48-A-c-4-4.tif
organisefiles["Gajlity"] = ["N-34-66-C-c-4-3", "N-34-77-A-b-1-4"]  # N-34-66-C-c-4-3

organisefiles["Jedrzejow"] = [
    "M-34-5-D-d-4-2",
    "M-34-6-A-d-2-2",
    "N-34-140-A-b-3-2",
    "N-34-140-A-b-4-2",
    "N-34-140-A-d-4-2",
    "N-34-140-A-d-3-4",
]  # N-34-140-A-d-4-2.tif
organisefiles["Zajeziorze"] = [
    "M-34-55-B-b-4-1",
    "M-34-56-A-b-1-4",
    "M-34-68-B-a-1-3",
    "M-34-32-B-b-1-3",
    "M-34-32-B-a-4-3",
]  # M-34-56-A-b-1-4.tif
organisefiles["Rokietnica"] = [
    "N-33-104-A-c-1-1",
    "N-33-130-A-d-3-3",
    "N-33-130-A-d-4-4",
    "N-33-119-C-c-3-3",
]  # N-33-130-A-d-4-4.tif


for name in organisefiles:
    sorted(organisefiles[name])

for name in organisefiles:
    makepath(name)

    I = len(organisefiles[name])
    XY = {}
    for token in organisefiles[name][0 : I // 2]:
        XY[token] = ("images/" + token + ".tif", "masks/" + token + ".tif")
    resizefile(path, XY, rootminiworld + name + "/test", 25)

    XY = {}
    for token in organisefiles[name][I // 2 :]:
        XY[token] = ("images/" + token + ".tif", "masks/" + token + ".tif")
    resizefile(path, XY, rootminiworld + name + "/train", 25)
