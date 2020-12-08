rm -r ../../__pycache__
rm -r build
mkdir build

echo "supervised color"
/data/anaconda3/bin/python -u train.py POTSDAM_lod0
/data/anaconda3/bin/python -u test.py  POTSDAM_lod0 | tee ../results/p2pcolor.txt

echo "supervised"
/data/anaconda3/bin/python -u train.py POTSDAM_lod0 normalize
/data/anaconda3/bin/python -u test.py  POTSDAM_lod0 normalize | tee ../results/p2p.txt


rm -r ../../__pycache__
