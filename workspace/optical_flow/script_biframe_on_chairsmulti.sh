#!/bin/bash

device=0

#ipython -- train_video_flow.py --nframes 7 \
#  --arch PWCDCNetStack_2by2 --bs 4 \
#  --savedir videoflow --expname debug \
#  --chairs --chairs-lr 0.0001 --test-sintel \
#  --nb-iter-per-epoch 16 --chairs-nb-epochs 30 \
#  --scheduler-step-indices 10 15 20 \
#  --scheduler-factors 0.5 \
#  --display-interval 1 \
#  --test-interval 1 --visu-visdom --nw 4

ipython -- train_video_flow.py --nframes 2 \
 --arch PWCDCNetStack_2by2 --len-seq-fixed \
 --savedir videoflow --expname PWCNet_2by2_biframe_on_chairsmulti \
 --chairs --chairs-lr 0.0001 --test-sintel \
 --nb-iter-per-epoch 1000 --chairs-nb-epochs 300 \
 --scheduler-step-indices 150 200 250 \
 --scheduler-factors 0.5 \
 --display-interval 40 --test-interval 5 \
 --visu-visdom --nw 4 --device $device

