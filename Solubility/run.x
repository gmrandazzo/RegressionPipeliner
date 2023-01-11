python3 make_model.py dataset.nosalt.rdkit_dscriptors.csv target.nosalt.csv
rm emissions.csv
python3 make_model.py dataset.nosalt.rdkit_morgan_ecfp.csv target.nosalt.csv
rm emissions.csv

