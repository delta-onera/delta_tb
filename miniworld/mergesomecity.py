import os

rootminiworld = "/scratchf/miniworld/"

citiesTOmerge = [
    "Fordon",
    "Grzedy",
    "Predocin",
    "Rokietnica",
    "Zajeziorze",
    "Gajlity",
    "Jedrzejow",
    "Preczow",
]
outputname = "pologne"


os.makedirs(rootminiworld + name)
os.makedirs(rootminiworld + name + "/train")
os.makedirs(rootminiworld + name + "/test")

for flag in ["/train/", "/test/"]:
    NBTOT = 0
    for city in citiesTOmerge:
        path = rootminiworld + city + flag
        NB = 0
        while os.path.exists(path + str(NB) + "_x.png"):
            NB += 1

        if NB == 0:
            print("wrong path", path)
            quit()

        for i in range(NB):
            os.system(
                "cp "
                + path
                + str(i)
                + "_x.png "
                + rootminiworld
                + name
                + flag
                + str(NBTOT)
                + "_x.png"
            )
            os.system(
                "cp "
                + path
                + str(i)
                + "_y.png "
                + rootminiworld
                + name
                + flag
                + str(NBTOT)
                + "_y.png"
            )
            NBTOT += 1
