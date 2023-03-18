import os
import PIL
from PIL import Image
import numpy

l = os.listdir("build")
assert len(l) == 6
l = [name for name in l if name != "predictions"]

l = sorted(l)  # sorted by perf

ll = []
for i in range(5):
    ll.append(os.listdir("build/" + l[i]))
for i in range(5):
    ll[i] = sorted(ll[i])
for i in range(4):
    assert ll[0] == ll[i + 1]

ll = ll[0]


def bestvotes(votes):
    assert votes.shape[0] == 5

    merged = dict()
    for i in range(5):
        if votes[i] in merged:
            merged[votes[i]] += 1
        else:
            merged[votes[i]] = 1

    # majority
    l = [v for (v, nb) in merged.items() if nb >= 3]
    if l != []:
        return l[0]

    # 5 != -> return the choose of the best model
    if len(merged) == 5:
        return votes[0]

    # necessarily 1,1,1,2 so we return the 2 which agree
    if len(merged) == 4:
        l = [v for (v, nb) in merged.items() if nb == 2]
        return l[0]

    # only 1,2,2 is left (because 1,1,3 has already been done)
    if len(merged) == 3:
        l = [v for (v, nb) in merged.items() if nb == 2]
        if votes[0] in l:
            return votes[0]
        else:
            return votes[1]


for name in ll:
    output = numpy.uint8(numpy.zeros((512 * 512)))
    votes = numpy.zeros((len(l), 512 * 512))
    for k in range(5):
        tmp = PIL.Image.open("build/" + l[k] + "/" + name).convert("L").copy()
        votes[k, :] = (numpy.asarray(tmp)).flatten()
    votes = numpy.uint8(numpy.transpose(votes))

    # parallel for :-(
    for i in range(512 * 512):
        output[i] = bestvotes(votes[i])

    output = output.reshape(512, 512)
    output = PIL.Image.fromarray(output)
    output.save("build/predictions/" + name)


"""
    for i in range(512):
        for j in range(512):
            histo = numpy.zeros(12)
            for k in range(5):
                histo[votes[k][i][j]]+=1
            I = histo[histo>=3]
            if I.shape[0]!=0:
                output[i][j] = I[0]
                break
            
            I = histo[histo==2]
            if I.shape[0]==1:
                output[i][j] = I[0]
                break
            if I.shape[0]==2:
                if histo[histo==votes[0][i][j]].shape[0]==2:
                 

quit()

import torch
import os

nbPixelPerClass = torch.zeros(13) #TABLE I of the paper
nbPixelPerClass[0] = 1670300028
nbPixelPerClass[1] = 1636681162
nbPixelPerClass[2] = 2836695330 
nbPixelPerClass[3] =  741261583
nbPixelPerClass[4] = 1034960677 
nbPixelPerClass[5] =  540883511
nbPixelPerClass[6] = 3061129298
nbPixelPerClass[7] = 1408504416 
nbPixelPerClass[8] =  665099230
nbPixelPerClass[9] = 3798549882 
nbPixelPerClass[10] =2061788310 
nbPixelPerClass[11] = 720090325 
nbPixelPerClass[12] = 117147576

testratio = torch.zeros(13) #fig 6
testratio[0] = 8.59
testratio[1] = 7.33
testratio[2] = 14.96 
testratio[3] =  4.35
testratio[4] = 5.98 
testratio[5] =  2.39
testratio[6] = 13.91
testratio[7] = 6.9 
testratio[8] =  3.86
testratio[9] = 22.15 
testratio[10] =7.03 
testratio[11] = 2.25 
testratio[12] = 0.29

models = os.listdir("build")

nbpred = torch.zeros(13,len(models))
"""
