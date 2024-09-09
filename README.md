# LasRCA

Artifacts accompanying LasRCA, a framework for one-shot root cause analysis in cloud-native systems that leverages the collaboration of the LLM and the small classifier. 

## Requirements

### Dependencies

````
cd ./code
pip install -r requirements.txt
````

### Our Test Sandbox

- Intel(R) Xeon(R) Gold 6226R CPU
- 128GB RAM
- NVIDIA GeForce RTX 4090 GPU
- Ubuntu 20.04.6 LTS
- Docker version 19.03.12
- Python 3.10 (Docker image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel)

## Folder Structure

The model and temp data needs to be downloaded from https://drive.google.com/drive/folders/1GK2GrJ6zxMI_cOCm91sWt38d_pUIQb0q?usp=sharing.

Then the ``model`` and ``temp_data`` folders need be placed following the structure of the working folder:
````
.
├── README.md
├── code                                          
│   ├── data_filter                                   preprocess data
│   ├── llm                                           generate LLM requests, save responses and process to fault labels
│   ├── pretrain                                      pretrain TS2Vec models for services, pods, and nodes (If there is no need to conduct experiments in motivation, it can be ignored)
│   ├── shared_util                                   some basic util functions like load/save pickle files, set deterministic seeds, etc.
│   ├── tasks                                         main experiments
│   │   ├── models                                    components of the small classifier
│   │   ├── base.py                                   base classes for RCA
│   │   ├── base_fault_type_classification.py         semi-supervised learning strategy and some other strategies for RCA
│   │   ├── dataset.py                                load dataset
│   │   ├── eval.py                                   evaluation metrics
│   │   ├── fine_tuning_fault_type_classification.py  pretrain and fine-tuning strategy for RCA
│   │   ├── las_rca.py                                main experiments of LasRCA
│   │   └── util                                      some util functions for different strategies
│   ├── pretrain.sh                                   shell for pretrain TS2Vec models
│   ├── reproduce.sh                                  shell for reproducibility
│   ├── reproduce.log                                 the execution log for reproducibility
│   └── requirements.txt
├── model                                             saved model data for reproduction (place the extracted ``model`` folder here)
└── temp_data                                         saved temp data for reproduction (place the extracted ``temp_data`` folder here)
````

## Quick Start / Reproducibility

### Prerequisites

1. Prepare the Python packages in ``requirements.txt``.
2. Unzip ``model.zip`` and ``temp_data.zip``. Then move the folders ``model`` and ``temp_data`` to the above correct situation.

Note: Although the path is configurable in our code, we strongly recommend placing the code on a path consistent with our experiment (``/workspace/project/working/2024/``), as this may avoid additional path parameter settings.

### Simple Result Checking

````
cd ./code
bash reproduce.sh
````

## Raw Data

Since the raw data is too big, we list the link here, help for downloading. The link is https://competition.aiops-challenge.com/home/competition/1496398526429724760. (Sometimes the page may be crashed, please visit https://www.bizseer.com/index.php?m=content&c=index&a=show&catid=25&id=83 for simple introduction).

## Running from Scratch based on Raw Data

Note: Although the path is configurable in our code, we strongly recommend placing the code on a path consistent with our experiment (``/workspace/project/working/2024/``), as this may avoid additional path parameter settings.

### Preprocess Raw Data

First put the raw data downloaded from the above link into the path described in ``code/data_filter/CCF_AIOps_challenge_2022/config/dataset_config.py``. (The default path is ``/workspace/dataset/2022_CCF_AIOps_challenge/``). The folder structure is as follows:

````
/workspace/dataset/2022_CCF_AIOps_challenge
├── test_data
├── training_data_normal
└── training_data_with_faults
````

Then run the following Python scripts to preprocess raw data.

````
cd ./code
python ./data_filter/CCF_AIOps_challenge_2022/dao/ground_truth_dao.py
python ./data_filter/CCF_AIOps_challenge_2022/dao/topology_dao.py
python ./data_filter/CCF_AIOps_challenge_2022/dao/metric_dao.py
python ./data_filter/CCF_AIOps_challenge_2022/dao/log_dao.py
python ./data_filter/CCF_AIOps_challenge_2022/dao/trace_dao.py
python ./data_filter/CCF_AIOps_challenge_2022/service/time_interval_label_generator.py
python ./data_filter/CCF_AIOps_challenge_2022/service/ent_edge_index_generator.py
python ./data_filter/CCF_AIOps_challenge_2022/service/metric_generator.py
python ./data_filter/CCF_AIOps_challenge_2022/service/log_generator.py
python ./data_filter/CCF_AIOps_challenge_2022/service/trace_generator.py
python ./data_filter/CCF_AIOps_challenge_2022/service/dataset_generator.py
````

Then you will find a new folder named ``temp_data`` that is side by side with the ``code``.

### Pretrain on TS2Vec (Optional)

If you want to replicate the pretraining and fine-tuning experiments mentioned in the paper, you need to first pretrain the embedding. Run the following command:

````
cd ./code
bash pretrain.sh
````

### Attempts on Fine-tuning or Semi-supervised Learning

1. If you want to run this step, you need to first run the ``Pretrain on TS2Vec``.
2. Uncomment the 311-314 lines in ``./code/tasks/base_fault_type_classification.py``, and comment the 316-338 lines in ``./code/tasks/base_fault_type_classification.py``.
3. Uncomment the 299-301 lines in ``./code/tasks/fine_tuning_fault_type_classification.py``, and comment the 303-324 lines in ``./code/tasks/fine_tuning_fault_type_classification.py``.
4. then run the following scripts:
````
python ./tasks/fine_tuning_fault_type_classification.py (for Fine-tuning)
python ./tasks/base_fault_type_classification.py (for Semi-supervised Learning)
````

### One-Shot Small Classifier Training

1. Uncomment the 267-268 lines in ``./code/tasks/las_rca.py``, and comment the 275-297 lines in ``./code/tasks/las_rca.py``.
2. Configure the parameter ``k`` in the 211 line to 0.
3. Run the command ``python ./tasks/las_rca.py``.
4. You need to set seed to 405-409 (line 210). Then comment the 267-268 lines, and uncomment 272-273 lines.

### LLM Fault Labeling

Before conducting LLM Fault Labeling, it is necessary to consider the correct settings of the following parameters:
- ``k`` in the 7 line of ``code/llm/common.py``: The iteration of LLM fault labeling. We use the value ``0`` as default. Therefore, LLM fault labeling is only executed once by default.
- ``api_key`` in the 8 line of ``code/llm/common.py``: Due to the high cost of GPT's API, we cannot disclose our api_key. You can switch api_key to your own account.

Then run the following scripts:
````
python ./llm/first_stage.py
python ./llm/second_stage.py
python ./llm/third_stage.py
python ./llm/summarization_stage.py
python ./llm/update_dataset.py
````

### One-Shot Small Classifier Retraining

1. Uncomment the 267-270 lines in ``./code/tasks/las_rca.py``, and comment the 275-297 lines in ``./code/tasks/las_rca.py``.
2. Configure the parameter ``k`` in the 211 line to 1, and the parameter ``k`` in the 212 line to "GPT4".
3. Run the command ``python ./tasks/las_rca.py``.
