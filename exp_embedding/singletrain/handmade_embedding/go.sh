rm -r ../../__pycache__
rm -r build
mkdir build

echo "transfert"
/data/anaconda3/bin/python -u train.py VAIHINGEN*
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/allv2p.txt

/data/anaconda3/bin/python -u train.py BRUGES*
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/allb2p.txt

/data/anaconda3/bin/python -u train.py VAIHINGEN BRUGES
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/vb2p.txt

/data/anaconda3/bin/python -u train.py VAIHINGEN* BRUGES*
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/allvb2p.txt

echo "semi-supervised"
/data/anaconda3/bin/python -u train.py VAIHINGEN BRUGES POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/vbp2p.txt

/data/anaconda3/bin/python -u train.py VAIHINGEN* BRUGES* POTSDAM
/data/anaconda3/bin/python -u test.py  POTSDAM | tee build/allvbp2p.txt


rm -r ../../__pycache__
