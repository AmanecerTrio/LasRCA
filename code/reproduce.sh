#！/bin/bash
# cd /workspace/project/working/2024/LasRCA/code
# nohup bash reproduce.sh > reproduce.log 2>&1 &

echo "Pretrain and Fine-tuning"
python ./tasks/fine_tuning_fault_type_classification.py

echo "Semi-supervised Learning"
python ./tasks/base_fault_type_classification.py

echo "Original One-Shot"
python ./tasks/las_rca.py --k 0

echo "LasRCA GPT4"
python ./tasks/las_rca.py --k 1 --labeling_strategy GPT4

echo "LasRCA SRE Corrections"
python ./tasks/las_rca.py --k 1 --labeling_strategy sre_corrections

echo "LasRCA random labeling 12 samples"
python ./tasks/las_rca.py --k 1 --labeling_strategy random_labeling_12

echo "LasRCA random labeling 24 samples"
python ./tasks/las_rca.py --k 1 --labeling_strategy random_labeling_24

echo "LasRCA random labeling 36 samples"
python ./tasks/las_rca.py --k 1 --labeling_strategy random_labeling_36

echo "LasRCA random labeling 48 samples"
python ./tasks/las_rca.py --k 1 --labeling_strategy random_labeling_48

echo "LasRCA random labeling 60 samples"
python ./tasks/las_rca.py --k 1 --labeling_strategy random_labeling_60
