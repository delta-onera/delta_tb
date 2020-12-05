rm -r ../../__pycache__
rm -r build
mkdir build

echo "transfert"
/data/anaconda3/bin/python -u train.py AIRS*
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/aa2p.txt

/data/anaconda3/bin/python -u train.py VAIHINGEN*
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/vv2p.txt

/data/anaconda3/bin/python -u train.py AIRS* VAIHINGEN* BRUGES*
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/vvbbaa2p.txt

echo "semi-supervised"
/data/anaconda3/bin/python -u train.py AIRS VAIHINGEN BRUGES POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/avbp2p.txt

/data/anaconda3/bin/python -u train.py AIRS* VAIHINGEN* BRUGES* POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/aavvbbp2p.txt


rm -r ../../__pycache__
