rm -r ../../__pycache__
rm -r build
mkdir build

/home/achanhon/anaconda3/bin/python -u train.py AIRS | tee build/airs_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  AIRS | tee build/airs_test.txt

/home/achanhon/anaconda3/bin/python -u train.py VAIHINGEN | tee build/vaihingen_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  VAIHINGEN | tee build/vaihingen_test.txt

/home/achanhon/anaconda3/bin/python -u train.py VAIHINGEN_lod0 | tee build/vaihingenlod0_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  VAIHINGEN_lod0 | tee build/vaihingenlod0_test.txt

/home/achanhon/anaconda3/bin/python -u train.py VAIHINGEN_lod0 normalize | tee build/vaihingenlod0normalize_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  VAIHINGEN_lod0 normalize | tee build/vaihingenlod0normalize_test.txt

/home/achanhon/anaconda3/bin/python -u train.py POTSDAM | tee build/potsdam_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  POTSDAM | tee build/potsdam_test.txt

/achanhon/anaconda3/bin/python -u train.py POTSDAM_lod0 | tee build/potsdamlod0_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  POTSDAM_lod0 | tee build/potsdamlod0_test.txt

/achanhon/anaconda3/bin/python -u train.py POTSDAM_lod0 normalize | tee build/potsdamlod0_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  POTSDAM_lod0 normalize | tee build/potsdamlod0_test.txt

/home/achanhon/anaconda3/bin/python -u train.py BRUGES | tee build/bruges_train.txt
/home/achanhon/anaconda3/bin/python -u test.py  BRUGES | tee build/burges_test.txt

rm -r ../../__pycache__
