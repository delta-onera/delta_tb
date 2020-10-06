rm -r ../../__pycache__
rm -r build
mkdir build

/data/anaconda3/bin/python -u vaihingen_train.py | tee build/vaihingen_train.txt
/data/anaconda3/bin/python -u vaihingen_test.py  | tee build/vaihingen_test.txt

/data/anaconda3/bin/python -u potsdam_train.py | tee build/potsdam_train.txt
/data/anaconda3/bin/python -u potsdam_test.py  | tee build/potsdam_test.txt

/data/anaconda3/bin/python -u dfc_train.py | tee build/dfc_train.txt
/data/anaconda3/bin/python -u dfc_test.py  | tee build/dfc_test.txt

/data/anaconda3/bin/python -u airs_train.py | tee build/airs_train.txt
/data/anaconda3/bin/python -u airs_test.py  | tee build/airs_test.txt

rm -r ../../__pycache__
