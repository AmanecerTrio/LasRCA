import os
import sys
import pandas as pd

sys.path.append('/workspace/project/working/2024/LasRCA/code')

import numpy as np
import json
import pickle
from tqdm import tqdm

from shared_util.common import *
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator

# cd /workspace/project/working/2024/LasRCA/code
# nohup python -u ./data_filter/CCF_AIOps_challenge_2022/service/trace_generator.py > ./data_filter/CCF_AIOps_challenge_2022/service/trace_generator.log 2>&1 &


class TraceGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def load_raw_trace(self):
        return self.raw_trace_dao.load_trace_csv()

    def calculate_trace_statistic(self):
        statistic_dict = dict()
        data_dict = dict()

        raw_data = self.load_raw_trace()
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['span_features']
                    for feature_name in feature_df.keys():
                        if feature_name == 'timestamp':
                            continue
                        exact_feature_name = TraceGenerator.extract_entity_feature_name(feature_name)
                        if exact_feature_name not in data_dict.keys():
                            statistic_dict[exact_feature_name] = 0
                            data_dict[exact_feature_name] = []
                        data_dict[exact_feature_name].extend(raw_data[dataset_type][date][cloud_bed]['span_features'][feature_name].tolist())

        for feature_name in statistic_dict.keys():
            trace_data = data_dict[feature_name]
            median = np.nanmedian(trace_data)
            percentile_1 = np.nanpercentile(trace_data, 1)
            percentile_99 = np.nanpercentile(trace_data, 99)
            q1 = np.nanpercentile(trace_data, 25)
            q3 = np.nanpercentile(trace_data, 75)
            mean = np.nanmean(trace_data)
            std = np.nanstd(trace_data)
            valid_ratio = (np.count_nonzero(~np.isnan(trace_data))) / len(list(trace_data))

            statistic_dict[feature_name] = {
                'mean': mean,
                'std': std,
                'percentile_1': percentile_1,
                'q1': q1,
                'median': median,
                'q3': q3,
                'percentile_99': percentile_99,
                'valid_ratio': valid_ratio
            }

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/trace'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/statistic.json', 'w') as f:
            json.dump(statistic_dict, f, indent=2)

    def z_score_trace_data(self):
        raw_data = self.load_raw_trace()

        file_dict = self.config.data_dict['file']
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/trace/statistic.json', 'r') as f:
            statistic_dict = json.load(f)

        for dataset_type, dataset_detail_dict in file_dict.items():
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['span_features']
                    for feature_name in feature_df.keys():
                        if feature_name == 'timestamp':
                            continue
                        exact_feature_name = TraceGenerator.extract_entity_feature_name(feature_name)
                        raw_trace_feature_data = raw_data[dataset_type][date][cloud_bed]['span_features'][feature_name]

                        iqr = statistic_dict[exact_feature_name]['q3'] - statistic_dict[exact_feature_name]['q1']
                        median = statistic_dict[exact_feature_name]['median']

                        if iqr != 0:
                            update_trace_feature_data = (raw_trace_feature_data - median) / iqr
                            for i in range(len(update_trace_feature_data)):
                                raw_data[dataset_type][date][cloud_bed]['span_features'].loc[i, feature_name] = update_trace_feature_data[i]

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/trace'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/all_trace_features.pkl', 'wb') as f:
            pickle.dump(raw_data, f)

    def slice_trace_features(self):
        result_dict = dict()

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/trace'
        os.makedirs(folder, exist_ok=True)
        raw_trace_features = self.load_raw_trace()
        with open(f'{folder}/all_trace_features.pkl', 'rb') as f:
            trace_features = pickle.load(f)
        container_list = self.config.data_dict['setting']['metric']['pod_order']
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dict[dataset_type] = dict()
            for date in dataset_detail_dict['date']:
                result_dict[dataset_type][date] = dict()
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    result_dict[dataset_type][date][cloud_bed] = {
                        'span_features': dict(),
                        'raw_span_features': dict()
                    }
                    feature_df = trace_features[dataset_type][date][cloud_bed]['span_features']
                    raw_feature_df = raw_trace_features[dataset_type][date][cloud_bed]['span_features']
                    for i in range(len(container_list)):
                        temp_dict, raw_temp_dict = {
                            'timestamp': feature_df['timestamp']
                        }, {
                            'timestamp': raw_feature_df['timestamp']
                        }
                        for feature_type in ['<intensity>', '<duration>']:
                            for feature_direction in ['upstream', 'current', 'downstream']:
                                feature_name = f'{feature_type}; cmdb_id: {container_list[i]}; type: {feature_direction}'
                                temp_dict[feature_name] = feature_df[feature_name]
                                raw_temp_dict[feature_name] = raw_feature_df[feature_name]
                        result_dict[dataset_type][date][cloud_bed]['span_features'][container_list[i]] = pd.DataFrame(temp_dict)
                        result_dict[dataset_type][date][cloud_bed]['raw_span_features'][container_list[i]] = pd.DataFrame(raw_temp_dict)

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/trace'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/all_trace.pkl', 'wb') as f:
            pickle.dump(result_dict, f)

    @staticmethod
    def extract_entity_feature_name(feature_name):
        cmdb_id = feature_name.split(';')[1].replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '')
        return f'{feature_name.split(";")[0]};{cmdb_id};{feature_name.split(";")[2]}'

    def get_all_trace(self) -> dict:
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/trace'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/all_trace.pkl', 'rb') as f:
            trace = pickle.load(f)
            return trace


if __name__ == '__main__':
    trace_generator = TraceGenerator()
    trace_generator.calculate_trace_statistic()
    trace_generator.z_score_trace_data()
    trace_generator.slice_trace_features()
    test = trace_generator.get_all_trace()
