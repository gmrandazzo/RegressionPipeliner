# Kinase Chemogenomic Set

Source: [https://www.mdpi.com/1422-0067/22/2/566](https://www.mdpi.com/1422-0067/22/2/566)

Every target represents the %inhibition at 1 uM of substrate concentration.
These values are acquired using the [KINOMEscan technology](https://www.discoverx.com/technologies-platforms/competitive-binding-technology/kinomescan-technology-platform).

# Results

To reproduce the entire pipeline please execute run.x

## Premise

These results are calculated using two molecular representations:
* ECFP fingerprint
* RDKit molecular descriptors

Train/test/validation sets are equal at each run of the ML method and thus comparable.

For each model on each kinase target we calculate as metrics:
- Mean square error
- Mean absolute error
- r2
- emission (kW): the energy necessary to train the model.

These scores are calculated using ONLY the validation set. Hence, no train/test values are considered.

## To the results

Most of the model generated here are poor regressors.
Some kinase target shows good R2 > 0.7. Those ones can be used for inference.

### ECFP results

TODO

### Descriptors results

TODO

### Carbon footprint analysis

These results show the average precision-recall area (AVG PR) versus the receiver operating characteristic area (ROC AUC) for all kinases targets per ML method.
The size of every spot represents the energy "impact" utilized to train the ML model.

| ECFP RESULTS  | DESC RESULTS  |
| ------------- |:-------------:|
| ![KCGS Results11](https://raw.githubusercontent.com/gmrandazzo/RegressionPipeliner/main/KinaseChemoGenomicSet/FinalResults/rdkit_ecfp_table_results.png) | ![KCGS Results12](https://raw.githubusercontent.com/gmrandazzo/RegressionPipeliner/main/KinaseChemoGenomicSet/FinalResults/rdkit_desc_table_results.png)     |

DNN (neural network) in both molecular descriptions requires a lot of energy compared to the other ML methods.
Moreover, a simpler method, such as TODO


Detailed results about the external validation set performances per kinase target
can be consulted on the respective directories [RDKitDescriptorsResults](https://raw.githubusercontent.com/gmrandazzo/RegressionPipeliner/main/KinaseChemoGenomicSet/FinalResults/RDKitDescriptorsResults) and [RDKitECFPMorganFP](https://raw.githubusercontent.com/gmrandazzo/RegressionPipeliner/main/KinaseChemoGenomicSet/FinalResults/RDKitECFPMorganFP).


# Interesting articles

- [Deep Learning Enhancing Kinome-Wide Polypharmacology Profiling: Model Construction and Experiment Validation](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00855)

- [Global Analysis of Deep Learning Prediction Using Large-Scale In-House Kinome-Wide Profiling Data](https://pubs.acs.org/doi/10.1021/acsomega.2c00664)
