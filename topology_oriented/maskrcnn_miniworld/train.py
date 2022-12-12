import os
import torch
import torchvision
import miniworld

assert torch.cuda.is_available()

print("load data")
dataset = miniworld.getMiniworld("/train/")

print("define model")
net = miniworld.MaskRCNN()
net = net.cuda()


print("train")
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
printloss = torch.zeros(1).cuda()
stats = torch.zeros((2, 2)).cuda()
batchsize = 8
nbbatchs = 300000
dataset.start()

for i in range(nbbatchs):
    x, y = dataset.getBatch(batchsize)
    x, D = x.cuda(), torch.ones(y.shape).cuda()

    z = net(x=x, y=y)

    loss = z["loss_objectness"] + z["loss_mask"] + z["loss_rpn_box_reg"]
    loss = loss + z["loss_classifier"] * 0.001 + z["loss_box_reg"] * 0.1

    with torch.no_grad():
        printloss += loss.clone().detach()
        y, z = y.cuda(), net(x=x)
        z = (z[:, 1, :, :] > z[:, 0, :, :]).clone().detach().float()
        stats += miniworld.confusion(y, z, D)

        if i < 10:
            print(i, "/", nbbatchs, printloss)
        if i < 1000 and i % 100 == 99:
            print(i, "/", nbbatchs, printloss / 100)
            printloss = torch.zeros(1).cuda()
        if i >= 1000 and i % 300 == 299:
            print(i, "/", nbbatchs, printloss / 300)
            printloss = torch.zeros(1).cuda()

        if i % 1000 == 999:
            torch.save(net, "build/model.pth")
            perf = miniworld.perf(stats)
            stats = torch.zeros((2, 2)).cuda()
            print(i, "perf", perf)
            if perf[0] > 95:
                os._exit(0)

    if i > nbbatchs * 0.1:
        loss = loss * 0.5
    if i > nbbatchs * 0.2:
        loss = loss * 0.5
    if i > nbbatchs * 0.5:
        loss = loss * 0.5
    if i > nbbatchs * 0.8:
        loss = loss * 0.5

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
    optimizer.step()

os._exit(0)
