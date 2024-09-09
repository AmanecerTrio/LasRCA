import os

from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator
from data_filter.CCF_AIOps_challenge_2022.service.metric_generator import MetricGenerator
from data_filter.CCF_AIOps_challenge_2022.service.trace_generator import TraceGenerator
from data_filter.CCF_AIOps_challenge_2022.service.log_generator import LogGenerator
from data_filter.CCF_AIOps_challenge_2022.service.ent_edge_index_generator import EntEdgeIndexGenerator
import pandas as pd
import pickle


class DatasetGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.meta_data = {
            'modal_types': ['metric', 'trace', 'log'],
            "ent_types": ['node', 'service', 'pod'],
            'ent_names': self.all_entity_list,
            'ent_type_index': {
                "node": (0, 6),
                "service": (6, 16),
                "pod": (16, 56)
            },
            'ent_fault_type_index': {
                "node": (0, 6),
                "service": (6, 15),
                "pod": (6, 15)
            },
            'ent_feature_num': {
                "node": {
                    "metric": 0,
                    "trace": 0,
                    "log": 0
                },
                "service": {
                    "metric": 0,
                    "trace": 0,
                    "log": 0
                },
                "pod": {
                    "metric": 0,
                    "trace": 0,
                    "log": 0
                }
            },
            'o11y_names': {
                "node": {
                    "metric": [],
                    "trace": [],
                    "log": []
                },
                "service": {
                    "metric": [],
                    "trace": [],
                    "log": []
                },
                "pod": {
                    "metric": [],
                    "trace": [],
                    "log": []
                }
            },
            "fault_type_list": self.fault_type_list,
            "en_fault_type_list": self.en_fault_type_list,
            "ent_edge_index": EntEdgeIndexGenerator().get_ent_edge_index()
        }

    def generate_final_data(self):
        metric = MetricGenerator().get_all_metric() if 'metric' in self.meta_data['modal_types'] else None
        trace = TraceGenerator().get_all_trace() if 'trace' in self.meta_data['modal_types'] else None
        log = LogGenerator().get_all_log() if 'log' in self.meta_data['modal_types'] else None
        raw_metric = self.raw_metric_dao.load_metric_csv() if 'metric' in self.meta_data['modal_types'] else None

        file_dict = self.config.data_dict['file']
        data_dict, raw_data_dict = dict(), dict()
        for dataset_type, dataset_detail_dict in file_dict.items():
            data_dict[dataset_type], raw_data_dict[dataset_type] = dict(), dict()
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    data_dict[dataset_type][f'{date}/{cloud_bed}'] = dict()
                    raw_data_dict[dataset_type][f'{date}/{cloud_bed}'] = dict()
                    for ent_type in self.meta_data['ent_types']:
                        data_dict[dataset_type][f'{date}/{cloud_bed}'][ent_type] = dict()
                        raw_data_dict[dataset_type][f'{date}/{cloud_bed}'][ent_type] = dict()
                        ent_type_index = self.meta_data['ent_type_index'][ent_type]
                        for ent_name in self.all_entity_list[ent_type_index[0]:ent_type_index[1]]:
                            data_dict[dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name] = dict()
                            raw_data_dict[dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name] = dict()
                            if ent_type == 'pod':
                                temp, raw_temp = metric, raw_metric
                                if metric is not None:
                                    metric_data, raw_metric_data = metric[dataset_type][date][cloud_bed], raw_metric[dataset_type][date][cloud_bed]
                                    temp = pd.concat([metric_data['container'][ent_name].set_index('timestamp'), metric_data['istio'][ent_name].set_index('timestamp')], axis=1, join='inner').reset_index()
                                    raw_temp = pd.concat([raw_metric_data['container'][ent_name].set_index('timestamp'), raw_metric_data['istio'][ent_name].set_index('timestamp')], axis=1, join='inner').reset_index()
                                    if self.meta_data['ent_feature_num'][ent_type]['metric'] == 0:
                                        columns = temp.columns[temp.columns != "timestamp"]
                                        self.meta_data['ent_feature_num'][ent_type]['metric'] = len(columns)
                                        self.meta_data['o11y_names'][ent_type]['metric'] = list(columns)
                                if trace is not None:
                                    if temp is not None:
                                        temp = pd.concat([temp.set_index('timestamp'), trace[dataset_type][date][cloud_bed]['span_features'][ent_name].set_index('timestamp')], axis=1, join='inner').reset_index()
                                        raw_temp = pd.concat([raw_temp.set_index('timestamp'), trace[dataset_type][date][cloud_bed]['raw_span_features'][ent_name].set_index('timestamp')], axis=1, join='inner').reset_index()
                                    else:
                                        temp = trace[dataset_type][date][cloud_bed]['span_features'][ent_name]
                                        raw_temp = trace[dataset_type][date][cloud_bed]['raw_span_features'][ent_name]
                                    if self.meta_data['ent_feature_num'][ent_type]['trace'] == 0:
                                        columns = trace[dataset_type][date][cloud_bed]['span_features'][ent_name].columns[trace[dataset_type][date][cloud_bed]['span_features'][ent_name].columns != "timestamp"]
                                        self.meta_data['ent_feature_num'][ent_type]['trace'] = len(columns)
                                        self.meta_data['o11y_names'][ent_type]['trace'] = list(columns)
                                if log is not None:
                                    if temp is not None:
                                        temp = pd.concat([temp.set_index('timestamp'), log[dataset_type][date][cloud_bed]['log_features'][ent_name].set_index('timestamp')], axis=1, join='inner').reset_index()
                                        raw_temp = pd.concat([raw_temp.set_index('timestamp'), log[dataset_type][date][cloud_bed]['raw_log_features'][ent_name].set_index('timestamp')], axis=1, join='inner').reset_index()
                                    else:
                                        temp = log[dataset_type][date][cloud_bed]['log_features'][ent_name]
                                        raw_temp = log[dataset_type][date][cloud_bed]['raw_log_features'][ent_name]
                                    if self.meta_data['ent_feature_num'][ent_type]['log'] == 0:
                                        columns = log[dataset_type][date][cloud_bed]['log_features'][ent_name].columns[log[dataset_type][date][cloud_bed]['log_features'][ent_name].columns != "timestamp"]
                                        self.meta_data['ent_feature_num'][ent_type]['log'] = len(columns)
                                        self.meta_data['o11y_names'][ent_type]['log'] = list(columns)

                                data_dict[dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name] = temp
                                raw_data_dict[dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name] = raw_temp
                            else:
                                data_dict[dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name] = metric[dataset_type][date][cloud_bed][ent_type][ent_name] if metric else None
                                raw_data_dict[dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name] = raw_metric[dataset_type][date][cloud_bed][ent_type][ent_name] if raw_metric else None
                                if metric is not None and self.meta_data['ent_feature_num'][ent_type]['metric'] == 0:
                                    columns = metric[dataset_type][date][cloud_bed][ent_type][ent_name].columns[metric[dataset_type][date][cloud_bed][ent_type][ent_name].columns != "timestamp"]
                                    self.meta_data['ent_feature_num'][ent_type]['metric'] = len(columns)
                                    self.meta_data['o11y_names'][ent_type]['metric'] = list(columns)

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/final'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/{"_".join(self.meta_data["modal_types"])}.pkl', 'wb') as f:
            pickle.dump({
                'meta_data': self.meta_data,
                'data': data_dict,
                'raw_data': raw_data_dict
            }, f)


if __name__ == '__main__':
    dataset_generator = DatasetGenerator()
    dataset_generator.generate_final_data()
