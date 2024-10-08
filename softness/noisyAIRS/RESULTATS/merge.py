import numpy

RESOLUTION = ["30", "50", "70", "100"]
BRUIT = ["nonoise","pm1image", "pm1translation", "hallucination"]
METHODE = ["base", "base-bord", "base+bord"]
MARGE = ["0", "1", "2", "bordonly"]


def readresults(resolution, bruit, methode, marge):
    assert resolution in RESOLUTION
    assert bruit in BRUIT
    assert methode in METHODE
    assert marge in MARGE

    path = "AIRS_" + bruit + "_" + resolution + "_" + methode + "_" + marge + ".csv"
    tmp = numpy.loadtxt(path)
    return tmp[0] / 10


print("check file exists")
tmp = 0
for bruit in BRUIT:
    for marge in MARGE:
        for methode in METHODE:
            for resolution in RESOLUTION:
                tmp += readresults(resolution, bruit, methode, marge)

print("done")

s = "\t"
for resolution in RESOLUTION:
    for bruit in BRUIT:
        s = s + bruit + resolution + "\t"
s = s + "\n"
for methode in METHODE:
    s = s + methode + "\t"
    for resolution in RESOLUTION:
        for bruit in BRUIT:
            s = s + str(readresults(resolution, bruit, methode, "0")) + "\t"
    s = s + "\n"

print(s)
