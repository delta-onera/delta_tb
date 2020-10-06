rm -r ../../__pycache__
rm -r build
mkdir build

/data/anaconda3/bin/python -u airs_train.py | tee build/airs_train.txt
/data/anaconda3/bin/python -u airs_test.py  | tee build/airs_test.txt

rm -r ../../__pycache__
