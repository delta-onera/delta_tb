rm -r ../../__pycache__
rm -r build
mkdir build

for town in 'Angers' 'Caen' 'Cherbourg' 'Lille_Arras_Lens_Douai_Henin'  'Marseille_Martigues' 'Nice' 'Rennes' 'Vannes' 'Brest' 'Calais_Dunkerque'  'Clermont-Ferrand' 'LeMans' 'Lorient' 'Nantes_Saint-Nazaire' 'Quimper' 'Saint-Brieuc'
do
    /home/jcastillo/anaconda3/bin/python -u tinyminifrance_train.py $town | tee build/tinyminifrance_train$town.txt
    /home/jcastillo/anaconda3/bin/python -u tinyminifrance_test.py $town | tee build/tinyminifrance_test$town.txt
done

rm -r ../../__pycache__
