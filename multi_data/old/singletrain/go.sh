echo "############################ different training settings -- all test on potsdam ############################"

cd embedding_color
sh go.sh
cd ..

cd finetune_embedding
sh go.sh
cd ..

cd baseline
sh go.sh
cd ..

cd handmade_embedding
sh go.sh
cd ..

cd embedding
sh go.sh
cd ..




