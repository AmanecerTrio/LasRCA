import os
import json


class DataConfig:
    def __init__(self):
        self.data_dict = dict()
        self.param_dict = dict()

        self.set_data_dict()
        self.set_param_dict()

    def set_data_dict(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metric_setting.json')) as f:
            metric_setting_dict = json.load(f)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'service_knowledge.json')) as f:
            service_knowledge_dict = json.load(f)

        data_base_path = '/workspace/dataset/2022_CCF_AIOps_challenge/'
        self.data_dict = {
            'file': {
                'normal': {
                    'base_folder': data_base_path + 'training_data_normal',
                    'date': ['2022-03-19'],
                    'cloud_bed': ['cloudbed-1', 'cloudbed-2', 'cloudbed-3']
                },
                'train_valid': {
                    'base_folder': data_base_path + 'training_data_with_faults',
                    'date': ['2022-03-20', '2022-03-21', '2022-03-24'],
                    'cloud_bed': ['cloudbed-1', 'cloudbed-2', 'cloudbed-3']
                },
                'test': {
                    'base_folder': data_base_path + 'test_data',
                    'date': ['2022-05-01', '2022-05-03', '2022-05-05', '2022-05-07', '2022-05-09'],
                    'cloud_bed': ['cloudbed'],
                }
            },
            'ground_truth': {
                'train_valid': data_base_path + 'training_data_with_faults/groundtruth',
                'test': data_base_path + 'test_data/groundtruth-csv'
            },
            'setting': {
                'metric': metric_setting_dict,
                'service_knowledge': service_knowledge_dict
            }
        }

    def set_param_dict(self):
        data_base_path = '/workspace/project/working/2024/LasRCA'
        self.param_dict = {
            'logging': {
                'level': 'DEBUG'
            },
            'temp_data_storage': f'{data_base_path}/temp_data/2022_CCF_AIOps_challenge'
        }
