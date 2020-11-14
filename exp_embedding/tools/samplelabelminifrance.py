

fewshot = {
    "Angers" : ['49-2013-0415-6705-LA93-0M50-E080_976_3529.tif', '49-2013-0455-6715-LA93-0M50-E080_3790_623.tif', '49-2013-0455-6715-LA93-0M50-E080_3326_5614.tif'],
    "Caen" : ['14-2012-0460-6910-LA93-0M50-E080_44_5340.tif', '14-2012-0470-6910-LA93-0M50-E080_8563_1409.tif', '14-2012-0470-6910-LA93-0M50-E080_826_2069.tif'],
    "Cherbourg" : ['50-2012-0375-6950-LA93-0M50-E080_5570_4215.tif', '50-2012-0380-6960-LA93-0M50-E080_8312_2519.tif', '50-2012-0380-6960-LA93-0M50-E080_8177_4112.tif'],
    "Lille_Arras_Lens_Douai_Henin" : ['62-2012-0705-7025-LA93-0M50-E080_3217_3343.tif', '62-2012-0700-7045-LA93-0M50-E080_8716_7689.tif', '62-2012-0710-7020-LA93-0M50-E080_7832_1151.tif'],
    "Marseille_Martigues" : ['13-2014-0895-6290-LA93-0M50-E080_3064_4174.tif', '13-2014-0925-6295-LA93-0M50-E080_8229_8797.tif', '13-2014-0925-6295-LA93-0M50-E080_3135_2527.tif'],
    "Nice" : ['06-2014-1045-6335-LA93-0M50-E080_587_2693.tif', '06-2014-1050-6315-LA93-0M50-E080_8716_2124.tif', '06-2014-1050-6315-LA93-0M50-E080_7046_4892.tif'],
    "Rennes" : ['35-2012-0365-6810-LA93-0M50-E080_5394_8656.tif', '35-2012-0370-6800-LA93-0M50-E080_7968_1161.tif', '35-2012-0370-6800-LA93-0M50-E080_7399_3835.tif'],
    "Vannes" : ['56-2013-0265-6750-LA93-0M50-E080_4710_3525.tif', '56-2013-0275-6760-LA93-0M50-E080_5104_3948.tif', '56-2013-0285-6750-LA93-0M50-E080_7324_8776.tif'],
    "Brest" : ['29-2012-0150-6855-LA93-0M50-E080_143_4164.tif', '29-2012-0170-6830-LA93-0M50-E080_3404_8749.tif', '29-2012-0170-6830-LA93-0M50-E080_6071_214.tif'],
    "Calais_Dunkerque" : ['62-2012-0635-7090-LA93-0M50-E080_3585_7658.tif', '62-2012-0640-7090-LA93-0M50-E080_1843_8933.tif', '62-2012-0640-7095-LA93-0M50-E080_801_7892.tif'],
    "Clermont-Ferrand" : ['63-2013-0725-6540-LA93-0M50-E080_6401_8352.tif', '63-2013-0715-6520-LA93-0M50-E080_4271_5295.tif', '63-2013-0735-6510-LA93-0M50-E080_8925_8046.tif'],
    "LeMans" : ['72-2013-0495-6775-LA93-0M50-E080_7274_1374.tif', '72-2013-0505-6765-LA93-0M50-E080_58_8678.tif', '72-2013-0505-6765-LA93-0M50-E080_5108_3755.tif'],
    "Lorient" : ['56-2013-0235-6760-LA93-0M50-E080_8657_4763.tif', '56-2013-0235-6775-LA93-0M50-E080_3651_2565.tif', '56-2013-0240-6775-LA93-0M50-E080_3776_2484.tif'],
    "Nantes_Saint-Nazaire" : ['44-2013-0365-6695-LA93-0M50-E080_1766_624.tif', '44-2013-0375-6705-LA93-0M50-E080_8961_5214.tif', '44-2013-0375-6705-LA93-0M50-E080_4847_2827.tif'],
    "Quimper" : ['29-2012-0185-6790-LA93-0M50-E080_1391_8685.tif', '29-2012-0185-6790-LA93-0M50-E080_2616_7329.tif', '29-2012-0185-6795-LA93-0M50-E080_8466_3065.tif'],
    "Saint-Brieuc" : ['22-2012-0285-6840-LA93-0M50-E080_612_8355.tif', '22-2012-0285-6840-LA93-0M50-E080_7779_5785.tif', '22-2012-0285-6840-LA93-0M50-E080_2545_954.tif']
}

import os
os.system("rm -r build")
os.system("mkdir build")

for town in fewshot:
    for name in fewshot[town]:
         os.system("cp /data01/PUBLIC_DATASETS/MiniFrance/tmFrance/BDORTHO/"+town+"/"+name+" build/"+name)

