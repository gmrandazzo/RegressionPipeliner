# Solubility Dataset

Source: [https://codeocean.com/capsule/8848590/tree/v1](https://codeocean.com/capsule/8848590/tree/v1)
Ref: [https://www.nature.com/articles/s41597-019-0151-1](https://www.nature.com/articles/s41597-019-0151-1)

Solubility expressed in logS units

# Results

To reproduce the entire pipeline please execute run.x

Molecular descriptors (best model R2 validation 0.86) perform better than ECFP (best model R2 validation 0.72).
All models here are good models. DNN/CatBoost/XGBoost and SVR methods perform better
than others. DNN requires 13 times more energy than CatBoost and
50 times compared to SVR (kernel: rbf).

| ECFP RESULTS  | DESC RESULTS  |
| ------------- |:-------------:|
| ![ECFP Results](https://raw.githubusercontent.com/gmrandazzo/RegressionPipeliner/main/Solubility/dataset.nosalt.rdkit_morgan_ecfp.png) | ![Desc Results](https://raw.githubusercontent.com/gmrandazzo/RegressionPipeliner/main/Solubility/dataset.nosalt.rdkit_dscriptors.png)     |
