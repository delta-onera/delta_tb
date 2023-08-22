import torch
import dataloader
import sys

assert torch.cuda.is_available()

print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()
if len(sys.argv) > 1:
    net.half()

print("load data")
dataset = dataloader.FLAIR2("val")

print("val")
stats = torch.zeros((13, 13)).cuda()
with torch.no_grad():
    for name in dataset.paths:
        x, s, y = dataset.get(name)
        x, s, y = x.cuda(), s.cuda(), y.cuda()

        if len(sys.argv) == 1:
            z = net(x.unsqueeze(0), s.unsqueeze(0))
        else:
            z = net(x.unsqueeze(0).half(), s.unsqueeze(0).half())

        _, z = z[0].max(0)
        stats += dataloader.confusion(y, z)

    print(stats)
    perf = dataloader.perf(stats)
    print("perf", perf)
