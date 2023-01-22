import yaml

with open("/home/achanhon/Bureau/tmp/PRED_061713_old.tif",'r') as texte:
    a=yaml.load(texte, Loader=yaml.loader.SafeLoader)
print(a)
