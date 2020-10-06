rm -r ../../__pycache__
rm -r build
mkdir build

/data/anaconda3/bin/python -u dfc_train.py | tee build/dfc_train.txt
/data/anaconda3/bin/python -u dfc_test.py  | tee build/dfc_test.txt

rm -r ../../__pycache__
