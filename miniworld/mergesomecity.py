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

os.makedirs(rootminiworld + outputname)
os.makedirs(rootminiworld + outputname + "/train")
os.makedirs(rootminiworld + outputname + "/test")

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
            pathin = path + str(i)
            pathout = rootminiworld + outputname + flag + str(NBTOT)
            os.system("cp " + pathin + "_x.png " + pathout + "_x.png")
            os.system("cp " + pathin + "_y.png " + pathout + "_y.png")
            NBTOT += 1
