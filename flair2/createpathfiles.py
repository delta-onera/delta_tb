import os
import json
import numpy
import torch


def number6(i):
    s = str(i)
    while len(s) < 6:
        s = "0" + s
    return s


root = "/scratchf/CHALLENGE_IGN/FLAIR_2/"

with open(root + "flair-2_centroids_sp_to_patch.json") as fichier:
    coords = json.load(fichier)
    checkforget = set(coords.keys())
    # coords = {key: coords[key] for key in sorted(coords.keys())}

trainpaths = {}
trainfolder, trainsubfolder = os.listdir(root + "flair_aerial_train/"), []
for folder in trainfolder:
    subfolder = os.listdir(root + "flair_aerial_train/" + folder)
    trainsubfolder += [(folder + "/" + sub) for sub in subfolder]

for tmp in coords:
    num = int(tmp[4:-4])

    for folder in trainsubfolder:
        tmp = "flair_aerial_train/" + folder + "/img/IMG_"
        if os.path.exists(root + tmp + number6(num) + ".tif"):
            trainpaths[num] = {}
            checkforget.remove("IMG_" + number6(num) + ".tif")
            trainpaths[num]["image"] = tmp + number6(num) + ".tif"
            tmp = "flair_labels_train/" + folder + "/msk/MSK_"
            trainpaths[num]["label"] = tmp + number6(num) + ".tif"
            assert os.path.exists(root + trainpaths[num]["label"])

            tmp = "flair_sen_train/" + folder + "/sen/"
            l = os.listdir(root + tmp)

            l = [s for s in l if "data.npy" in s]
            assert len(l) == 1
            trainpaths[num]["sen"] = tmp + l[0]
            tmp = numpy.load(root + trainpaths[num]["sen"])

            a, b = coords["IMG_" + number6(num) + ".tif"]
            if 20 <= a <= tmp.shape[2] and 20 <= b <= tmp.shape[3]:
                trainpaths[num]["coord"] = (a - 20, b - 20)
            else:
                print(a, b, tmp.shape)
                quit()

torch.save(trainpaths, root + "alltrainpaths.pth")

print("THE SAME WITH TEST !!!!!!!!")
trainpaths = {}
trainfolder, trainsubfolder = os.listdir(root + "flair_aerial_train/"), []
for folder in trainfolder:
    subfolder = os.listdir(root + "flair_aerial_train/" + folder)
    trainsubfolder += [(folder + "/" + sub) for sub in subfolder]

for tmp in coords:
    num = int(tmp[4:-4])

    for folder in trainsubfolder:
        tmp = "flair_2_aerial_test/" + folder + "/img/IMG_"
        if os.path.exists(root + tmp + number6(num) + ".tif"):
            trainpaths[num] = {}
            checkforget.remove("IMG_" + number6(num) + ".tif")
            trainpaths[num]["image"] = tmp + number6(num) + ".tif"

            tmp = "flair_2_sen_test/" + folder + "/sen/"
            l = os.listdir(root + tmp)

            l = [s for s in l if "data.npy" in s]
            assert len(l) == 1
            trainpaths[num]["sen"] = tmp + l[0]
            tmp = numpy.load(root + trainpaths[num]["sen"])

            a, b = coords["IMG_" + number6(num) + ".tif"]
            if 20 <= a <= tmp.shape[2] and 20 <= b <= tmp.shape[3]:
                trainpaths[num]["coord"] = (a - 20, b - 20)
            else:
                print(a, b, tmp.shape)
                quit()

torch.save(trainpaths, root + "alltestpaths.pth")

print("checkforget", checkforget)
