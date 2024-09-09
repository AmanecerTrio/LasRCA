import os
import sys

sys.path.append('/workspace/project/working/2024/LasRCA/code')

import numpy as np
import json
from shared_util.common import *
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator

# cd /workspace/project/working/2024/LasRCA/code
# nohup python -u ./data_filter/CCF_AIOps_challenge_2022/service/metric_generator.py > ./data_filter/CCF_AIOps_challenge_2022/service/metric_generator.log 2>&1 &


class MetricGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def load_raw_metric(self):
        return self.raw_metric_dao.load_metric_csv()

    def calculate_common_statistic(self):
        common_statistic_dict = dict()
        data_dict = dict()

        raw_data = self.load_raw_metric()
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    resource_type_list = ['node', 'container', 'service', 'istio']
                    for resource_type in resource_type_list:
                        if resource_type not in data_dict.keys():
                            common_statistic_dict[resource_type] = dict()
                            data_dict[resource_type] = dict()
                        entity_list = self.raw_metric_dao.get_entity_list(resource_type)
                        for entity in entity_list:
                            merged_entity = MetricGenerator.merge_entity(resource_type, entity)
                            if merged_entity not in data_dict[resource_type].keys():
                                common_statistic_dict[resource_type][merged_entity] = dict()
                                data_dict[resource_type][merged_entity] = dict()
                            metric_name_list = raw_data[dataset_type][date][cloud_bed][resource_type][entity].keys()
                            for metric_name in metric_name_list:
                                if metric_name == 'timestamp':
                                    continue
                                if metric_name not in data_dict[resource_type][merged_entity].keys():
                                    common_statistic_dict[resource_type][merged_entity][metric_name] = 0
                                    data_dict[resource_type][merged_entity][metric_name] = []
                                metric_data = raw_data[dataset_type][date][cloud_bed][resource_type][entity][metric_name]
                                metric_data = MetricGenerator.diff_metric(metric_name, metric_data.tolist())
                                data_dict[resource_type][merged_entity][metric_name].extend(metric_data)  # .tolist())

        for resource_type, metric_dict in common_statistic_dict.items():
            for entity, entity_metric_dict in metric_dict.items():
                for metric_name in entity_metric_dict.keys():
                    metric_data = data_dict[resource_type][entity][metric_name]
                    common_statistic_dict[resource_type][entity][metric_name] = MetricGenerator.calculate_statistic(metric_data)

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/metric'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/common_statistic.json', 'w') as f:
            json.dump(common_statistic_dict, f, indent=2)

    def z_score_metric_data(self):
        raw_data = self.load_raw_metric()

        file_dict = self.config.data_dict['file']
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/metric/common_statistic.json', 'r') as f:
            common_statistic_dict = json.load(f)
        for dataset_type, dataset_detail_dict in file_dict.items():
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    # resource_type_list = ['node', 'container', 'service', 'istio']
                    resource_type_list = ['istio']
                    for resource_type in resource_type_list:
                        entity_list = self.raw_metric_dao.get_entity_list(resource_type)
                        for entity in entity_list:
                            merged_entity = MetricGenerator.merge_entity(resource_type, entity)
                            metric_name_list = raw_data[dataset_type][date][cloud_bed][resource_type][entity].keys()
                            for metric_name in metric_name_list:
                                if metric_name == 'timestamp':
                                    continue
                                raw_metric_data = raw_data[dataset_type][date][cloud_bed][resource_type][entity][metric_name]
                                raw_metric_data = np.array(MetricGenerator.diff_metric(metric_name, raw_metric_data))
                                common_mean = common_statistic_dict[resource_type][merged_entity][metric_name]['mean']
                                common_std = common_statistic_dict[resource_type][merged_entity][metric_name]['std']
                                raw_metric_data = np.array(raw_metric_data)
                                if not (np.isnan(common_mean) or np.isnan(common_std) or common_std == 0):
                                    update_metric_data = (raw_metric_data - common_mean) / common_std
                                    update_metric_data[np.isnan(update_metric_data)] = 0
                                for i in range(update_metric_data.shape[0]):
                                    raw_data[dataset_type][date][cloud_bed][resource_type][entity].loc[i, metric_name] = update_metric_data[i]

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/metric'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/all_metric.pkl', 'wb') as f:
            pickle.dump(raw_data, f)

    @staticmethod
    def calculate_statistic(metric_data):
        median = np.nanmedian(metric_data)
        percentile_1 = np.nanpercentile(metric_data, 1)
        percentile_99 = np.nanpercentile(metric_data, 99)
        q1 = np.nanpercentile(metric_data, 25)
        q3 = np.nanpercentile(metric_data, 75)
        mean = np.nanmean(metric_data)
        std = np.nanstd(metric_data)
        clip_data = np.clip(metric_data, percentile_1, percentile_99)
        clip_mean = np.nanmean(clip_data)
        clip_std = np.nanstd(clip_data)
        valid_ratio = (np.count_nonzero(~np.isnan(metric_data))) / len(list(metric_data))

        return {
            'clip_mean': clip_mean,
            'clip_std': clip_std,
            'percentile_1': percentile_1,
            'q1': q1,
            'median': median,
            'q3': q3,
            'percentile_99': percentile_99,
            'valid_ratio': valid_ratio,
            'mean': mean,
            'std': std
        }

    @staticmethod
    def merge_entity(resource_type, entity):
        if resource_type == 'container' or resource_type == 'istio':
            entity = entity.replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '')
        elif resource_type == 'node':
            entity = entity.split('-')[0]
        return entity

    @staticmethod
    def diff_metric(metric_name, metric_data):
        diff_name_list = [
            "Memory/system.mem.used",
            "Memory/system.mem.real.used",
            "Memory/system.mem.usable",
            "Disk/system.disk.free",
            "Disk/system.disk.used",
            "Memory/kpi_container_memory_usage_MB",
            "Memory/kpi_container_memory_working_set_MB",
            "Memory/kpi_container_memory_rss",
            "Memory/kpi_container_memory_mapped_file"
            "Disk/kpi_container_fs_reads_MB",
            "Disk/kpi_container_fs_usage_MB",
            "Disk/kpi_container_fs_writes_MB",
            "Thread/kpi_container_threads",
        ]
        if metric_name in diff_name_list:
            metric_data = np.array(metric_data)
            metric_data = np.diff(metric_data)
            metric_data = np.append(metric_data, metric_data[-1])
        return metric_data

    def get_all_metric(self) -> dict:
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/metric'
        with open(f'{folder}/all_metric.pkl', 'rb') as f:
            metric = pickle.load(f)
            return metric


if __name__ == '__main__':
    metric_generator = MetricGenerator()
    metric_generator.calculate_common_statistic()
    metric_generator.z_score_metric_data()
    metric_generator.get_all_metric()
