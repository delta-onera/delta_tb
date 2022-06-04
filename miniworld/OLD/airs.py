import os
import numpy
import PIL
from PIL import Image


def resize(image, label, resolutions):
    if resolutions[0] == resolutions[1]:
        return image, label
    coef = resolutions[0] / resolutions[1]
    size = (int(image.size[0] * coef), int(image.size[1] * coef))

    if image is not None:
        image = image.resize(size, PIL.Image.BILINEAR)
    if label is not None:
        label = label.resize(size, PIL.Image.NEAREST)
    return image, label


def resizeall(outpath, inpath, XY, resolutions):
    for i, name in enumerate(XY):
        x, y = XY[name]
        image = PIL.Image.open(inpath + "/" + x).convert("RGB").copy()
        label = PIL.Image.open(inpath + "/" + y).convert("L").copy()

        image, label = resize(image, label, resolutions)

        image.save(outpath + "/" + str(i) + "_x.png")
        label.save(outpath + "/" + str(i) + "_y.png")


root = "/scratchf/DATASETS/AIRS/trainval/"
os.system("rm -rf /scratchf/airs_multi_res")
os.system("mkdir /scratchf/airs_multi_res")
for resolution in [30, 50, 70, 100]:
    print(resolution)
    os.system("/scratchf/airs_multi_res/" + str(resolution))
    os.system("/scratchf/airs_multi_res/" + str(resolution) + "/christchurch/")
    path = "/scratchf/airs_multi_res/" + str(resolution) + "/christchurch/"

    os.system("mkdir " + path)
    os.system("mkdir " + path + "/train")
    os.system("mkdir " + path + "/test")

    for flag, flag2 in [("test", "val"), ("train", "train")]:
        XY = {}
        allname = os.listdir(root + flag2 + "/image")
        for name in allname:
            XY[name] = ["image/" + name[0:-4], "label/" + name[0:-4] + "_vis"]
            XY[name] = [XY[name][0] + ".tif", XY[name][1] + ".tif"]
        resizeall(path + flag, root + flag2, XY, (7.5, resolution))
