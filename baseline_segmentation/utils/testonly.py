import os
import sys
import PIL
from PIL import Image
import numpy

if len(sys.argv) == 1:
    print("wrong path")
    quit()
TODO = sys.argv[1] + "/"
print(TODO)

found = set(os.listdir(TODO))
names = [name for name in found if "y.png" in name]
names = [name[:-5] for name in names]
names = [name for name in names if name + "z.png" in found]

print(names)

cm = numpy.zeros((2, 2))

for name in names:
    y = PIL.Image.open(TODO + name + "y.png").convert("L").copy()
    z = PIL.Image.open(TODO + name + "z.png").convert("L").copy()

    y, z = numpy.asarray(y), numpy.asarray(z)

    z = numpy.int16(z > 125)
    D = numpy.absolute(y - 125) / 125
    y = numpy.int16(y > 125)

    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cm[a][b] += numpy.sum(numpy.int16(z == a) * numpy.int16(y == b) * D)

accu = 100.0 * (cm[0][0] + cm[1][1]) / (numpy.sum(cm) + 1)
iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
print(iou0 + iou1)
print(accu, 2 * iou0, 2 * iou1, cm)
