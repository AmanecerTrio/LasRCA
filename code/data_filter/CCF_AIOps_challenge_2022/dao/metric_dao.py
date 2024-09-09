import os
import sys

sys.path.append('/workspace/project/working/2024/LasRCA/code')

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

from data_filter.CCF_AIOps_challenge_2022.base.base_class import BaseClass
from shared_util.common import *
from shared_util.time_handler import TimeHandler

# cd /workspace/project/working/2024/LasRCA/code
# nohup python -W ignore -u ./data_filter/CCF_AIOps_challenge_2022/dao/metric_dao.py > ./data_filter/CCF_AIOps_challenge_2022/dao/metric_dao.log 2>&1 &


class RawMetricDao(BaseClass):
    def __init__(self):
        super().__init__()

    def process_node_metric(self, timestamp_list: list, data_base_path: str, result_base_path: str):
        node_list = self.config.data_dict['setting']['metric']['node_order']
        metric_info_dict = self.config.data_dict['setting']['metric']['related_metrics']['node']
        metric_df = pd.read_csv(glob.glob(f'{data_base_path}/node/*.csv')[0])
        metric_df.set_index(['timestamp', 'cmdb_id', 'kpi_name'], inplace=True, drop=False)

        result_base_path = f'{result_base_path}/node'
        os.makedirs(result_base_path, exist_ok=True)

        node_list_bar = tqdm(node_list)
        for node in node_list_bar:
            node_metric_dict = {
                'timestamp': timestamp_list
            }
            for metric_type, metric_name_list in metric_info_dict.items():
                for raw_metric_name in metric_name_list:
                    metric_name = f'{metric_type}/{raw_metric_name}'
                    node_metric_dict[metric_name] = []
                    for timestamp in timestamp_list:
                        if (timestamp, node, raw_metric_name) in metric_df.index:
                            node_metric_dict[metric_name].append(metric_df.loc[timestamp, node, raw_metric_name]['value'])
                        else:
                            node_metric_dict[metric_name].append(np.nan)
            node_df = pd.DataFrame(node_metric_dict)
            node_df.to_csv(f'{result_base_path}/{node}.csv', index=False)
            node_list_bar.set_description("Node metric csv generating".format(node))

    def process_container_metric(self, timestamp_list: list, data_base_path: str, result_base_path: str):
        pod_list = self.config.data_dict['setting']['metric']['pod_order']
        metric_info_dict = self.config.data_dict['setting']['metric']['related_metrics']['container']

        result_base_path = f'{result_base_path}/container'
        os.makedirs(result_base_path, exist_ok=True)

        container_metric_dict = dict()
        for pod in pod_list:
            container_metric_dict[pod] = {
                'timestamp': timestamp_list
            }

        for metric_type, metric_name_list in metric_info_dict.items():
            metric_name_bar = tqdm(metric_name_list)
            for raw_metric_name in metric_name_bar:
                metric_name = f'{metric_type}/{raw_metric_name}'

                metric_df = pd.read_csv(f'{data_base_path}/container/{raw_metric_name}.csv')
                metric_df.set_index(['timestamp', 'cmdb_id'], inplace=True, drop=False)

                for pod in pod_list:
                    container_metric_dict[pod][metric_name] = []
                    for timestamp in timestamp_list:
                        is_find = False
                        if timestamp in metric_df.index.levels[0]:
                            temp_df = metric_df.loc[timestamp]
                            temp_df = temp_df[temp_df.cmdb_id.str.contains(pod)]
                            if temp_df.shape[0] == 1:
                                container_metric_dict[pod][metric_name].append(temp_df.iloc[0]['value'])
                                is_find = True
                        if not is_find:
                            container_metric_dict[pod][metric_name].append(np.nan)
                metric_name_bar.set_description("Container metric csv generating".format(metric_name))

        for pod, metric_dict in container_metric_dict.items():
            metric_df = pd.DataFrame(metric_dict)
            metric_df.to_csv(f'{result_base_path}/{pod}.csv', index=False)

    def process_istio_metric(self, timestamp_list: list, data_base_path: str, result_base_path: str):
        pod_list = self.config.data_dict['setting']['metric']['pod_order']
        metric_info_dict = self.config.data_dict['setting']['metric']['related_metrics']['istio']

        result_base_path = f'{result_base_path}/istio'
        os.makedirs(result_base_path, exist_ok=True)

        istio_metric_dict = dict()
        for pod in pod_list:
            istio_metric_dict[pod] = {
                'timestamp': timestamp_list
            }

        for metric_type, metric_name_list in metric_info_dict.items():
            metric_name_bar = tqdm(metric_name_list)
            for raw_metric_name in metric_name_bar:
                metric_name = f'{metric_type}/{raw_metric_name}'
                metric_df = pd.read_csv(f'{data_base_path}/istio/{raw_metric_name}.csv')
                metric_df.set_index(['timestamp', 'cmdb_id', 'kpi_name'], inplace=True, drop=False)

                for pod in pod_list:
                    istio_metric_dict[pod][f'{metric_name}.source'] = []
                    istio_metric_dict[pod][f'{metric_name}.destination'] = []
                    for timestamp in timestamp_list:
                        if timestamp in metric_df.index.levels[0]:
                            temp_df = metric_df.loc[timestamp]
                            source_temp_df = temp_df[temp_df.cmdb_id.str.contains(f'{pod}.source')]

                            destination_temp_df = temp_df[temp_df.cmdb_id.str.contains(f'{pod}.destination')]

                            if source_temp_df.shape[0] > 0:
                                istio_metric_dict[pod][f'{metric_name}.source'].append(source_temp_df['value'].sum())
                            else:
                                istio_metric_dict[pod][f'{metric_name}.source'].append(np.nan)

                            if destination_temp_df.shape[0] > 0:
                                istio_metric_dict[pod][f'{metric_name}.destination'].append(
                                    destination_temp_df['value'].sum())
                            else:
                                istio_metric_dict[pod][f'{metric_name}.destination'].append(np.nan)
                        else:
                            istio_metric_dict[pod][f'{metric_name}.source'].append(np.nan)
                            istio_metric_dict[pod][f'{metric_name}.destination'].append(np.nan)

                metric_name_bar.set_description("Istio metric csv generating".format(metric_name))

        for pod, metric_dict in istio_metric_dict.items():
            metric_df = pd.DataFrame(metric_dict)
            metric_df.to_csv(f'{result_base_path}/{pod}.csv', index=False)

    def process_service_metric(self, timestamp_list: list, data_base_path: str, result_base_path: str):
        service_list = self.config.data_dict['setting']['metric']['service_order']
        metric_info_dict = self.config.data_dict['setting']['metric']['related_metrics']['service']
        metric_df = pd.read_csv(glob.glob(f'{data_base_path}/service/*.csv')[0])
        metric_df.set_index(['timestamp', 'service'], inplace=True, drop=False)

        result_base_path = f'{result_base_path}/service'
        os.makedirs(result_base_path, exist_ok=True)

        service_list_bar = tqdm(service_list)
        for service in service_list_bar:
            service_metric_dict = {
                'timestamp': timestamp_list
            }
            for metric_type, metric_name_list in metric_info_dict.items():
                for raw_metric_name in metric_name_list:
                    metric_name = f'{metric_type}/{raw_metric_name}'
                    service_metric_dict[f'{metric_name}.grpc'] = []
                    service_metric_dict[f'{metric_name}.http'] = []

                    for timestamp in timestamp_list:
                        if timestamp in metric_df.index.levels[0]:
                            temp_df = metric_df.loc[timestamp]
                            temp_df_grpc = temp_df[temp_df.service == f'{service}-grpc']
                            if temp_df_grpc.shape[0] == 1:
                                service_metric_dict[f'{metric_name}.grpc'].append(temp_df_grpc.iloc[0][raw_metric_name])
                            else:
                                service_metric_dict[f'{metric_name}.grpc'].append(np.nan)

                            temp_df_http = temp_df[temp_df.service == f'{service}-http']
                            if temp_df_http.shape[0] == 1:
                                service_metric_dict[f'{metric_name}.http'].append(temp_df_http.iloc[0][raw_metric_name])
                            else:
                                service_metric_dict[f'{metric_name}.http'].append(np.nan)
                        else:
                            service_metric_dict[f'{metric_name}.grpc'].append(np.nan)
                            service_metric_dict[f'{metric_name}.http'].append(np.nan)
            service_df = pd.DataFrame(service_metric_dict)
            service_df.to_csv(f'{result_base_path}/{service}.csv', index=False)
            service_list_bar.set_description("Service metric csv generating".format(service))

    def generate_metric_csv(self):
        file_dict = self.config.data_dict['file']
        result_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        # dataset_type/date/cloud_bed/resource_type/metric.csv
        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dataset_type_path = f'{result_base_path}/{dataset_type}'
            os.makedirs(result_dataset_type_path, exist_ok=True)
            for date in dataset_detail_dict['date']:
                result_date_path = f'{result_dataset_type_path}/{date}'
                os.makedirs(result_base_path, exist_ok=True)
                timestamp_list = [TimeHandler.datetime_to_timestamp(f'{date} 00:00:00') + i * 60 for i in
                                  range(0, 24 * 60)]
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    
                    result_cloud_bed_path = f'{result_date_path}/{cloud_bed}/raw_metric'
                    os.makedirs(result_cloud_bed_path, exist_ok=True)
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/metric'

                    for resource_type in ['node', 'container', 'service']:
                        logger.debug(f'Preprocessing metrics: dataset_type: {dataset_type}, date: {date}, '
                                          f'cloudbed: {cloud_bed}, resource_type: {resource_type}.')
                        eval(f'self.process_{resource_type}_metric(timestamp_list, data_base_path, result_cloud_bed_path)')

    def load_metric_csv(self):
        result_dict = dict()

        file_dict = self.config.data_dict['file']
        metric_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dict[dataset_type] = dict()
            metric_dataset_type_path = f'{metric_base_path}/{dataset_type}'
            for date in dataset_detail_dict['date']:
                result_dict[dataset_type][date] = dict()
                metric_date_path = f'{metric_dataset_type_path}/{date}'
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    result_dict[dataset_type][date][cloud_bed] = dict()
                    metric_cloud_bed_path = f'{metric_date_path}/{cloud_bed}/raw_metric'
                    for resource_type in ['node', 'container', 'service', 'istio']:
                        result_dict[dataset_type][date][cloud_bed][resource_type] = dict()
                        entity_list = self.get_entity_list(resource_type)
                        for entity in entity_list:
                            result_dict[dataset_type][date][cloud_bed][resource_type][entity] = pd.read_csv(f'{metric_cloud_bed_path}/{resource_type}/{entity}.csv').interpolate()
        return result_dict

    def get_entity_list(self, resource_type: str):
        entity_list = []
        if resource_type == 'node':
            entity_list = self.config.data_dict['setting']['metric']['node_order']
        elif resource_type == 'container' or resource_type == 'istio':
            entity_list = self.config.data_dict['setting']['metric']['pod_order']
        elif resource_type == 'service':
            entity_list = self.config.data_dict['setting']['metric']['service_order']
        return entity_list


if __name__ == '__main__':
    raw_metric_dao = RawMetricDao()
    raw_metric_dao.generate_metric_csv()
