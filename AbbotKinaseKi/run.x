#python3 make_model.py dataset.rdkit_dscriptors.csv ChEMBL_Navigating_the_Kinome_Ki.csv
#mkdir RDKitDescriptorsResults;
#mv *.json *.png RDKitDescriptorsResults

#python3 make_model.py dataset.morgan_ecfp.csv ChEMBL_Navigating_the_Kinome_Ki.csv
#mkdir RDKitECFPMorganFP;
#mv *.json *.png RDKitECFPMorganFP

rm rdkit_desc_table_results.csv
echo "Kin-Method,MSE,MAE,R2,Emission(kW)" >> rdkit_desc_table_results.csv
for i in RDKitDescriptorsResults/*.json; do python3 jsonresults2table.py "${i}" >> rdkit_desc_table_results.csv; done
rm rdkit_ecfp_table_results.csv
echo "Kin-Method,MSE,MAE,R2,Emission(kW)" >> rdkit_ecfp_table_results.csv
for i in RDKitECFPMorganFP/*.json; do python3 jsonresults2table.py "${i}" >> rdkit_ecfp_table_results.csv; done

python3 analyze_by_kinase_and_method.py rdkit_ecfp_table_results.csv rdkit_ecfp_table_results_by_kinase.png
python3 analyze_by_kinase_and_method.py rdkit_desc_table_results.csv rdkit_desc_table_results.png

python3 analyze_by_method.py rdkit_ecfp_table_results.csv rdkit_ecfp_table_results.png "Average Prediction Method Results using ECPF"
python3 analyze_by_method.py rdkit_desc_table_results.csv rdkit_desc_table_results.png "Average Prediction Method Results using Descriptors"

mkdir FinalResults
mv *.png FinalResults
