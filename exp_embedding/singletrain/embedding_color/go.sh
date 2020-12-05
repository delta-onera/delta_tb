rm -r ../../__pycache__
rm -r build
mkdir build

echo "semi-supervised embedding color"
/data/anaconda3/bin/python -u train_sum.py VAIHINGEN BRUGES POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/vbp2p_color_sum.txt

/data/anaconda3/bin/python -u train_sum.py VAIHINGEN* BRUGES* POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/allvbp2p_color_sum.txt

/data/anaconda3/bin/python -u train.py VAIHINGEN BRUGES POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/vbp2p_color_graph.txt

/data/anaconda3/bin/python -u train.py VAIHINGEN* BRUGES* POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/allvbp2p_color_graph.txt


rm -r ../../__pycache__
