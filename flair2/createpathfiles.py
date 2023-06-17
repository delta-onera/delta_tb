import os
import json
import numpy


def number6(i):
    s = str(i)
    while len(s) < 6:
        s = "0" + s
    return s


root = "/scratchf/CHALLENGE_IGN/FLAIR_2/"

with open(root + "flair-2_centroids_sp_to_patch.json") as fichier:
    coords = json.load(fichier)

trainpaths = {}

trainfolder, trainsubfolder = os.listdir(root + "flair_aerial_train/"), []
for folder in trainfolder:
    subfolder = os.listdir(root + "flair_aerial_train/" + folder)
    trainsubfolder += [(folder + "/" + sub) for sub in subfolder]

for tmp in coords:
    print(tmp)
    num = int(tmp[4:])
    print(num)
    trainpaths[num] = {}

    for folder in trainsubfolder:
        tmp = root + "flair_aerial_train/" + folder + "/img/IMG_"
        if os.path.exists(tmp + number6(num) + ".tif"):
            trainpaths[num]["image"] = tmp + number6(num) + ".tif"
            tmp = root + "flair_labels_train/" + folder + "/msk/MSK"
            assert os.path.exists(tmp + number6(num) + ".tif")
            trainpaths[num]["label"] = tmp + number6(num) + ".tif"

            tmp = "flair_sen_train/" + folder + "/sen/"
            l = os.listdir(root + tmp)
            l = [s for s in l if ".npy in s"]
            trainpaths[num]["sen"] = tmp + l

            tmp = numpy.load(root + trainpaths[num]["sen"])
            print(tmp)
            quit()

"""
testfolder = os.listdir(root + "flair_2_aerial_test/")
for folder in testfolder:
    subfolder = os.listdir(root + "flair_2_aerial_test/" + folder)
    testsubfolder += [(folder + "/" + sub) for sub in subfolder]
"""
