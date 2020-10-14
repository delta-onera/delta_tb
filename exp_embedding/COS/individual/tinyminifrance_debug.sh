rm -r ../../__pycache__
rm -r build
mkdir build

for town in  'LeMans'
do
    /home/jcastillo/anaconda3/bin/python -u tinyminifrance_train.py $town | tee build/tinyminifrance_train$town.txt
    /home/jcastillo/anaconda3/bin/python -u tinyminifrance_test.py $town | tee build/tinyminifrance_test$town.txt
done

rm -r ../../__pycache__
