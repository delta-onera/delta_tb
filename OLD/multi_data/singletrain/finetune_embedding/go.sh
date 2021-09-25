rm -r ../../__pycache__
rm -r build
mkdir build

echo "semi-supervised embedding with finetuning"
/data/anaconda3/bin/python -u train_sum.py AIRS VAIHINGEN BRUGES POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee ../results/avbp2p_finetune_sum.txt

/data/anaconda3/bin/python -u train.py AIRS VAIHINGEN BRUGES POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee ../results/avbp2p_finetune_graph.txt

rm -r ../../__pycache__
