import os
import PIL
from PIL import Image
import numpy
import torch

l = os.listdir("build/modelfort")
ll = os.listdir("build/modelspec")
assert sorted(l) == sorted(ll)
del ll

for name in l:
    spec = PIL.Image.open("build/modelspec/" + name).convert("L").copy()
    spec = torch.Tensor(numpy.asarray(spec))
    fort = PIL.Image.open("build/modelfort/" + name).convert("L").copy()
    fort = torch.Tensor(numpy.asarray(fort))

    with torch.no_grad():
        output = fort * (spec <= 10).float() + spec * (spec > 10).float()

    output = PIL.Image.fromarray(numpy.uint8(output.numpy()))
    output.save("build/predictions/" + name)
