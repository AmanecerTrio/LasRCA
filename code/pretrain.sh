#！/bin/bash
# cd /workspace/project/working/2024/LasRCA/code
# nohup bash pretrain.sh > pretrain.log 2>&1 &

echo "A_s/metric_trace_log start"
python ./pretrain/train.py --dataset A_s/metric_trace_log --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_trace_log.pkl
echo "A_s/metric_trace_log end"

echo "A_n/metric_trace_log start"
python ./pretrain/train.py --dataset A_n/metric_trace_log --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_trace_log.pkl
echo "A_n/metric_trace_log end"

echo "A_p/metric_trace_log start"
python ./pretrain/train.py --dataset A_p/metric_trace_log --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_trace_log.pkl --batch-size 16
echo "A_p/metric_trace_log end"

echo "A_s/metric_trace start"
python ./pretrain/train.py --dataset A_s/metric_trace --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_trace.pkl
echo "A_s/metric_trace start"

echo "A_n/metric_trace start"
python ./pretrain/train.py --dataset A_n/metric_trace --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_trace.pkl
echo "A_n/metric_trace end"

echo "A_p/metric_trace start"
python ./pretrain/train.py --dataset A_p/metric_trace --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_trace.pkl --batch-size 16
echo "A_p/metric_trace end"

echo "A_s/metric_log start"
python ./pretrain/train.py --dataset A_s/metric_log --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_log.pkl
echo "A_s/metric_log end"

echo "A_n/metric_log start"
python ./pretrain/train.py --dataset A_n/metric_log --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_log.pkl
echo "A_n/metric_log end"

echo "A_p/metric_log start"
python ./pretrain/train.py --dataset A_p/metric_log --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_log.pkl --batch-size 16
echo "A_p/metric_log end"

echo "A_p/trace_log start"
python ./pretrain/train.py --dataset A_p/trace_log --dataset_path /workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/trace_log.pkl --batch-size 16
echo "A_p/trace_log end"
