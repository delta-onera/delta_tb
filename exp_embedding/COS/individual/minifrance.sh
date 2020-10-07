rm -r ../../__pycache__
rm -r build
mkdir build

for town in 'Angers' 'Caen' 'Cherbourg' 'Lille_Arras_Lens_Douai_Henin'  'Marseille_Martigues' 'Nice' 'Rennes' 'Vannes' 'Brest' 'Calais_Dunkerque'  'Clermont-Ferrand' 'LeMans' 'Lorient' 'Nantes_Saint-Nazaire' 'Quimper' 'Saint-Brieuc'
do
    /data/anaconda3/bin/python -u minifrance_train.py $town | tee build/vaihingen_train_$town.txt
    /data/anaconda3/bin/python -u minifrance_test.py $town | tee build/vaihingen_test_$town.txt
done

rm -r ../../__pycache__
