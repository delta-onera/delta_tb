import os

root = "/scratchf/OSCD/"
folders = os.listdir(root)
folders = [f for f in folders if os.path.isdir(root + f)]
folders = [f for f in folders if os.path.exists(root + f + "/pair/img1.png")]
folders = [f for f in folders if os.path.exists(root + f + "/pair/img2.png")]
folders = [f for f in folders if os.path.exists(root + f + "/cm/cm.png")]

print(folders)
