import os
import PIL
from PIL import Image

root = "/scratchf/OSCD/"
folders = os.listdir(root)
folders = [f for f in folders if os.path.isdir(root + f)]
folders = [f for f in folders if os.path.exists(root + f + "/pair/img1.png")]
folders = [f for f in folders if os.path.exists(root + f + "/pair/img2.png")]
folders = [f for f in folders if os.path.exists(root + f + "/cm/cm.png")]

print(folders)

os.system("rm -r build")
os.system("mkdir build")

for path in folders:
    img1 = PIL.Image.open(root + path + "/pair/img1.png").convert("RGB").copy()
    img2 = PIL.Image.open(root + path + "/pair/img2.png").convert("RGB").copy()
    cm = PIL.Image.open(root + path + "/cm/cm.png").convert("L").copy()

    img1, img2, cm = numpy.asarray(img1), numpy.asarray(img2), numpy.asarray(cm)
    if img1.shape != img2.shape:
        print(path, img1.shape, img2.shape)
    if img1.shape[1:3] != cm.shape:
        print(path, img1.shape, cm.shape)
