rm -r ../../__pycache__
rm -r build
mkdir build

/data/anaconda3/bin/python -u potsdam_train.py | tee build/potsdam_train.txt
/data/anaconda3/bin/python -u potsdam_test.py  | tee build/potsdam_test.txt

rm -r ../../__pycache__
