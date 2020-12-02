rm -r ../../__pycache__
rm -r build
mkdir build

/home/achanhon/anaconda3/bin/python -u train.py VAIHINGEN | tee build/vaihingen_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  VAIHINGEN | tee build/vaihingen_test.txt

/home/achanhon/anaconda3/bin/python -u train.py POTSDAM | tee build/potsdam_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  POTSDAM | tee build/potsdam_test.txt

/home/achanhon/anaconda3/bin/python -u train.py BRUGES | tee build/bruges_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  BRUGES | tee build/burges_test.txt

rm -r ../../__pycache__
