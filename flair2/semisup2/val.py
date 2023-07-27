import torch
import dataloader

assert torch.cuda.is_available()

print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()

print("load data")
dataset = dataloader.FLAIR2("val")

print("val")
stats = torch.zeros((13, 13)).cuda()
with torch.no_grad():
    for name in dataset.paths:
        x, s, y = dataset.get(name)
        x, s, y = x.cuda(), s.cuda(), y.cuda()

        z = net(x.unsqueeze(0), s.unsqueeze(0))
        _, z = z[0].max(0)
        stats += dataloader.confusion(y, z)

    print(stats)
    perf = dataloader.perf(stats)
    print("perf", perf)
