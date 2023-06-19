import torch
import numpy


def compress(x):
    print(x.shape)
    xm = x.flatten(2).mean(2)
    print(xm.shape)
    x_ = x - xm.unsqueeze(-1).unsqueeze(-1)
    print(x_.shape)
    xv = torch.sqrt(((x_.flatten(2)) ** 2).mean(2))
    x_ = x_ / (xv.unsqueeze(-1).unsqueeze(-1) + 0.001)
    x_ = torch.clamp(x_, -3, 3)
    x__ = x / (x.flatten().max())
    x = torch.cat([x_, x__], dim=1)
    print(x.shape)

    tmp = x.flatten(1)
    print(tmp.shape)
    D = ((tmp.unsqueeze(0) - tmp.unsqueeze(1)) ** 2).mean(2)
    print(D.shape)

    T = [0, x.shape[0] // 2, x.shape[-1]]
    print(T)
    for t in range(19):
        d = D[T].transpose(0, 1)
        if t == 0:
            print(d.shape)
        d, _ = d.min(1)
        if t == 0:
            print(d.shape)
        i = d.argmax()
        if t == 0:
            print(i)
        T.append(i)

    x = x[T]
    assert x.shape[0:1] == (20, 20)
    return x


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

        # numpy.save(root + paths[i]["sen"], sentinel)
        quit()
