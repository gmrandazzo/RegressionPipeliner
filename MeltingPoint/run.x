gunzip dataset.rdkit_descriptors.csv.gz
python3 make_model.py dataset.rdkit_descriptors.csv target.csv
rm emissions.csv
gzip dataset.rdkit_descriptors.csv

gunzip dataset.rdkit_morgan_ecfp.csv.gz
python3 make_model.py dataset.rdkit_morgan_ecfp.csv target.csv
rm emissions.csv
gzip dataset.rdkit_morgan_ecfp.csv

#python3 make_model_runtime.py dataset.rdkit_descriptors.csv  target.csv runtime.csv
#python3 make_model_runtime.py dataset.rdkit_morgan_ecfp.csv target.csv runtime.csv

