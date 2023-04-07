import os
import PIL
from PIL import Image
import numpy

root = "/scratchf/OSCD/"
folders = os.listdir(root)
folders = [f for f in folders if os.path.isdir(root + f)]
folders = [f for f in folders if os.path.exists(root + f + "/pair/img1.png")]
folders = [f for f in folders if os.path.exists(root + f + "/pair/img2.png")]
folders = [f for f in folders if os.path.exists(root + f + "/cm/cm.png")]

print(folders)

os.system("rm -r build")
os.system("mkdir build")

size = 48
i = 0
for path in folders:
    img1 = PIL.Image.open(root + path + "/pair/img1.png").convert("RGB").copy()
    img2 = PIL.Image.open(root + path + "/pair/img2.png").convert("RGB").copy()
    cm = PIL.Image.open(root + path + "/cm/cm.png").convert("L").copy()

    img1, img2, cm = numpy.asarray(img1), numpy.asarray(img2), numpy.asarray(cm)
    if img1.shape != img2.shape:
        print(path, img1.shape, img2.shape)
    if img1.shape[0:2] != cm.shape:
        print(path, img1.shape, cm.shape)

    for row in range(0, cm.shape[0] - size - 2, size):
        for col in range(0, cm.shape[1] - size - 2, size):
            if cm[row : row + size, col : col + size].sum() > 0:
                continue
            crop1 = img1[row : row + size, col : col + size, :]
            crop2 = img2[row : row + size, col : col + size, :]
            crop1, crop2 = PIL.Image.fromarray(crop1), PIL.Image.fromarray(crop2)

            crop1.save("build/" + str(i) + "_1.png")
            crop2.save("build/" + str(i) + "_2.png")
            i += 1

            if i % 1000 == 999:
                print(i, path)

print("process terminated")
print(i)
