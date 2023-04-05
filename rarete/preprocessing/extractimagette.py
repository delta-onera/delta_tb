import os

root = "/scratchf/OSCD/"
folders = os.listdir(root)
folders = [f for f in folders if os.path.isdir(root + f)]
folders = [f for f in folders if os.path.isdir(root + f + "/pair")]
folders = [f for f in folders if os.path.isdir(root + f + "/cm")]

print(folders)
