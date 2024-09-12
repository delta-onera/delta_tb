import torch
import torchvision
import common

print("load data")
testset = common.S2Looking("test")
N = len(testset.l)
M = len(testset.badaffine.CANONICAL_TRANSFORM)


print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()

print("start testing")
with torch.no_grad():
    mse = 0
    for i in range(N):
        for j in range(M):
            img1, img2, cm = testset.get(i, j)
            img1, img2, cm = img1.cuda(), img2.cuda(), cm.cuda()
            pred = net(img1, img2)
            mse += float(((cm - pred) ** 2).sum())

    mse = mse / (M * N)
    print("mse on transformation matrix", mse)
    with open("build/results.txt", "a") as file_:
        file_.write(str(mse) + "\n")
