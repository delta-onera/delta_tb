import numpy

RESOLUTION = ["30", "50", "70", "100"]
BRUIT = ["pm1image", "nonoise", "pm1translation", "hallucination"]
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
for resolution in RESOLUTION:
    s=resolution+"\n\t"
    for bruit in BRUIT:
        s=s+bruit+"\t"
    s=s+"\n"
    for methode in METHODE:
        s = s+methode+"\t"
        for bruit in BRUIT:
            s = 
