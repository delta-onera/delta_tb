rm -r ../__pycache__
rm -r build
mkdir build

/data/anaconda3/bin/python -u doubledebugtrain.py | tee build/trainlog.txt
/data/anaconda3/bin/python -u doubledebugtest.py  | tee build/testlog.txt

rm -r ../__pycache__
