import torch
import numpy


def compress(x):
    xm = x.flatten(2).mean(2)
    x_ = x - xm.unsqueeze(-1).unsqueeze(-1)
    xv = torch.sqrt(((x_.flatten(2)) ** 2).mean(2))
    x_ = x_ / (xv.unsqueeze(-1).unsqueeze(-1) + 0.001)
    x_ = torch.clamp(x_, -4, 4)
    x__ = 4 * x / (x.flatten().max() + x.mean()) - 1
    x__ = torch.clamp(x__, -2, 2)
    x = torch.cat([x_, x__], dim=1).half().float()

    tmp = x.flatten(1)
    D = torch.zeros(x.shape[0], x.shape[0]).cuda()
    for i in range(x.shape[0]):
        D[i] = ((tmp - tmp[i].unsqueeze(0)) ** 2).mean(dim=1)

    T = [0, x.shape[0] - 1]
    for t in range(8):
        d = D[T].transpose(0, 1)
        d, _ = d.min(1)
        i = d.argmax()
        T = sorted(T + [int(i)])

    x = x[T]
    assert x.shape[0:2] == (10, 20)
    return x.half()


root = "/scratchf/CHALLENGE_IGN/FLAIR_2/"
l = ["alltestpaths.pth", "alltrainpaths.pth"]
for name in l:
    paths = torch.load(root + name)
    print(len(paths))
    for i in paths:
        sentinel = numpy.load(root + paths[i]["sen"])

        sentinel = torch.Tensor(numpy.float32(sentinel)).cuda()
        sentinel = compress(sentinel)
        sentinel = sentinel.cpu().numpy()

        numpy.save("/d/achanhon/tmp", sentinel)
        numpy.save(root + paths[i]["sen"], sentinel)

print("GOOOOOOOOOOOD")
