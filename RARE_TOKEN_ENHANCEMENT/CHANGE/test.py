import torch
import torchvision
import common

print("load data")
testset = common.S2Looking("test")
N = len(testset.l) * 4

print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()

print("start testing")
with torch.no_grad():
    res = torch.zeros(2, 2).cuda()
    for i in range(N):
        img1, img2, cm = testset.get(i)
        img1, img2, cm = img1.cuda(), img2.cuda(), cm.cuda()
        pred = net(img1, img2)
        _, pred = pred.max(0)

        for a, b in [(0, 0), (1, 1), (0, 1), (1, 0)]:
            res[a][b] += ((pred == a).float() * (cm == b).float()).sum()
    print("test cm", res)

    IoU1 = 1.0 * res[1][1] / (res[1][1] + res[0][1] + res[1][0] + 1)
    IoU2 = 1.0 * res[0][0] / (res[0][0] + res[0][1] + res[1][0] + 1)
    IoU1, IoU2 = float(IoU1), float(IoU2)
    IoU = (IoU1 + IoU2) / 2
    print("test IoU=", IoU, IoU1, IoU2)
    with open("build/results.txt", "a") as file_:
        out = str(IoU) + "\t" + str(IoU1) + "\t" + str(IoU2)
        file_.write(out + "\n")
