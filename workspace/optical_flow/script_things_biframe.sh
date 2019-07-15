#!/bin/bash

device=0
chairspath='/scratch/pgodet/ChairsMultiframe'
sintelpath='/data/FLOW_DATASET/sintel'


#ipython -- train_video_flow.py --nframes 7 \
#  --arch PWCDCNetStack_2by2 --bs 4 \
#  --savedir videoflow --expname debug \
#  --chairs --chairs-lr 0.0001 --test-sintel \
#  --nb-iter-per-epoch 16 --chairs-nb-epochs 30 \
#  --scheduler-step-indices 10 15 20 \
#  --scheduler-factors 0.5 \
#  --display-interval 1 \
#  --test-interval 1 --visu-visdom --nw 4

ipython -- train_supervised_flow.py --arch PWCDCNet_siamese \
 --pretrained /data/pgodet/resultats/pg_dcnn/chairsonly/train_pwcnet_gray_chairs_Sshort/checkpoint.pth.tar \
 --savedir PWCThings --expname things_final_pwcSshort \
 --things --things-lr 0.00001 --test-sintel \
 --sintel-path $sintelpath \
 --nb-iter-per-epoch 2000 --things-nb-epochs 250 \
 --things-scheduler-step-indices 100 150 200 \
 --things-scheduler-factors 0.5 \
 --display-interval 80 --test-interval 5 \
 --visu-visdom --nw 4 --device $device

ipython -- train_supervised_flow.py --arch PWCDCNet_siamese \
 --pretrained /data/pgodet/resultats/pg_dcnn/chairsonly/train_pwcnet_gray_chairs_Slong_bis/checkpoint.pth.tar \
 --savedir PWCThings --expname things_final_pwcSlong \
 --things --things-lr 0.00001 --test-sintel \
 --sintel-path $sintelpath \
 --nb-iter-per-epoch 2000 --things-nb-epochs 250 \
 --things-scheduler-step-indices 100 150 200 \
 --things-scheduler-factors 0.5 \
 --display-interval 80 --test-interval 5 \
 --visu-visdom --nw 4 --device $device

ipython -- train_supervised_flow.py --arch PWCDCNet_siamese \
 --pretrained /data/pgodet/resultats/pg_dcnn/chairsonly/train_pwcnet_gray_chairs_Sshort/checkpoint.pth.tar \
 --savedir PWCThings --expname things_final_kick1000_pwcSshort \
 --things --things-lr 0.00003 --test-sintel \
 --sintel-path $sintelpath \
 --nb-iter-per-epoch 1000 --things-nb-epochs 300 \
 --things-scheduler-step-indices 40 60 80 100 120 150 190 210 230 250 270 \
 --things-scheduler-factors 0.5 0.5 0.5 0.5 0.5 20 0.5 0.5 0.5 0.5 0.5\
 --display-interval 80 --test-interval 5 \
 --visu-visdom --nw 4 --device $device