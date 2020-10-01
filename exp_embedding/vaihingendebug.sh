rm -r build
mkdir build

/data/anaconda3/bin/python -u vaihingendebugtrain.py | tee trainlog.txt
/data/anaconda3/bin/python -u vaihingendebugtest.py  | tee testlog.txt
