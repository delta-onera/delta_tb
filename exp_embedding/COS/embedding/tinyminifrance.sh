rm -r ../../__pycache__
rm -r build
mkdir build

CUDA_VISIBLE_DEVICES=2,3 /home/jcastillo/anaconda3/bin/python -u tinyminifrance_train.py | tee build/tinyminifrance_train.txt
CUDA_VISIBLE_DEVICES=2,3 /home/jcastillo/anaconda3/bin/python -u tinyminifrance_test.py | tee build/tinyminifrance_test.txt

rm -r ../../__pycache__
