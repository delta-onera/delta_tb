import torch
import torchvision
import common

print("load data")
testset = common.EurosatSplit("test")
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()

print("start testing")
with torch.no_grad():
    cm = torch.zeros(10, 10).cuda()
    for inputs, targets in testloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        cm += common.confmat(targets, predicted)
    print("test cm", cm)

    total = float(torch.sum(cm))
    accuracy = float(torch.diagonal(cm).sum()) / total
    print("test accuracy=", accuracy)
    with open("build/results.txt", "a") as file_:
        file_.write(str(accuracy) + "\n")
