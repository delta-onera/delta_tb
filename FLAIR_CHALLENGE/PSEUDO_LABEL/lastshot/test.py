import torch
import torchvision
import dataloader
import PIL
import PIL.Image
import numpy

assert torch.cuda.is_available()

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()

print("load data")
dataset = dataloader.FLAIR("/scratchf/CHALLENGE_IGN/test/")

print("test")

with torch.no_grad():
    for i in range(len(dataset.paths)):
        if i % 100 == 99:
            print(i, "/", len(dataset.paths))
        x = dataset.getImage(i)
        x = x.cuda()

        z = net(x.unsqueeze(0))
        _, z = z[0].max(0)

        z = numpy.uint8(numpy.clip(z.cpu().numpy(), 0, 12))
        z = PIL.Image.fromarray(z)
        z.save("build/output/PRED_" + dataset.getName(i), compression="tiff_lzw")
