singletrain
    baseline : permet de lancer train ou test sur 1 jeux de données avec unet (car equivalent single case)
        nom du dataset (avec flag classes et objectif)
        pré processing (option)
    
    handmade_embedding : permet de lancer unet en train ou test sur un set de jeux de données
        listes des datasets forcément gris, normalisée, lod0     
        
    embedding : permet de lancer embedding en train ou test sur un set de jeux de données
        listes des datasets forcément gris, normalisée, lod0     

    embedding_color : embedding utilise les datasets prétraités exactement comme handmade_embedding alors que embedding_color permet d'utiliser les datasets natifs
        listes des datasets (avec flag classes et objectif)

sequentialtrain
    learning a representation and then applying on a new area
        
    
    
    
