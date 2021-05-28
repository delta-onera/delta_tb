rm -r ../../__pycache__
rm -r build
mkdir build

CUDA_VISIBLE_DEVICES=2 /home/jcastillo/anaconda3/bin/python -u wholetinyminifrance_train.py normalize | tee build/wholetinyminifrancenormalize_train.txt
CUDA_VISIBLE_DEVICES=2 /home/jcastillo/anaconda3/bin/python -u wholetinyminifrance_test.py normalize | tee build/wholetinyminifrancenormalize_test.txt

CUDA_VISIBLE_DEVICES=2 /home/jcastillo/anaconda3/bin/python -u wholetinyminifrance_train.py | tee build/wholetinyminifrance_train.txt
CUDA_VISIBLE_DEVICES=2 /home/jcastillo/anaconda3/bin/python -u wholetinyminifrance_test.py | tee build/wholetinyminifrance_test.txt

CUDA_VISIBLE_DEVICES=2 /home/jcastillo/anaconda3/bin/python -u wholetinyminifrance_train.py grey | tee build/wholetinyminifrancegrey_train.txt
CUDA_VISIBLE_DEVICES=2 /home/jcastillo/anaconda3/bin/python -u wholetinyminifrance_test.py grey | tee build/wholetinyminifrancegrey_test.txt

rm -r ../../__pycache__
