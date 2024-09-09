import os
import sys

import pandas as pd

sys.path.append('/workspace/project/working/2024/LasRCA/code')

import numpy as np
import json
import pickle

from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator


class LogGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def load_raw_log(self, log_type):
        return self.raw_log_dao.load_log_csv(log_type)

    @staticmethod
    def extract_entity_feature_name(feature_name):
        cmdb_id = feature_name.split(';')[0].replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '')
        return f'{cmdb_id};{feature_name.split(";")[1]}'

    def calculate_log_pattern_tf_idf(self):
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/log/log_patterns.json') as f:
            log_pattern_dict = json.load(f)

        result_dict = dict()
        temp_dict = dict()
        selected_pattern_dict = dict()
        for language, log_pattern_list in log_pattern_dict.items():
            result_dict[language] = {
                'analysis': {
                    'idf': dict(),
                    'max_tf': dict(),
                    'max_tf_idf': dict()
                },
                'raw_tf_idf': {
                    'tf': dict(),
                    'tf_idf': dict()
                }
            }
            temp_dict[language] = []
            for i in range(len(log_pattern_list)):
                temp_dict[language].append([])
            selected_pattern_dict[language] = []

        raw_data = self.load_raw_log('container')
        for dataset_type, dataset_detail_dict in self.config.data_dict['file'].items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['container_log_features']
                    language_service = self.config.data_dict['setting']['service_knowledge']['language_service']
                    for language, service_list in language_service.items():
                        log_pattern_list = log_pattern_dict[language]
                        service_list = language_service[language]
                        pod_list = []
                        for service in service_list:
                            pod_list.extend(LogGenerator.rename_service2pod(service))
                        for i in range(len(log_pattern_list)):
                            occurrence_list = feature_df.loc[:, [f'{pod}; <log pattern {i}>' for pod in pod_list]].apply(lambda x: x.sum(), axis=1).tolist()
                            temp_dict[language][i].extend(occurrence_list)
        for language in temp_dict.keys():
            sum_list = np.sum(temp_dict[language], axis=0)
            for i in range(len(temp_dict[language])):
                if not sum_list.any():
                    tf = temp_dict[language][i]
                else:
                    tf = np.true_divide(temp_dict[language][i], sum_list)
                idf = np.log(np.array(temp_dict[language][i]).shape[0] / (1 + np.count_nonzero(temp_dict[language][i])))
                result_dict[language]['analysis']['idf'][str(i)] = idf
                result_dict[language]['analysis']['max_tf'][str(i)] = np.sort(tf[~np.isnan(tf)])[-30:].mean()
                result_dict[language]['analysis']['max_tf_idf'][str(i)] = np.sort(tf[~np.isnan(tf)])[-30:].mean() * idf
                if result_dict[language]['analysis']['max_tf_idf'][str(i)] > 0.3:
                    selected_pattern_dict[language].append(i)

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/log'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/container_log_tf_idf_statistic.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

        with open(f'{folder}/container_log_selected_patterns.json', 'w') as f:
            json.dump(selected_pattern_dict, f, indent=2)

    def get_selected_log_pattern_features(self):
        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/log'
        with open(f'{folder}/container_log_selected_patterns.json') as f:
            selected_log_patterns = json.load(f)
        return selected_log_patterns

    def calculate_container_log_pattern_statistic(self):
        statistic_dict = dict()
        data_dict = dict()
        selected_log_pattern_features = []
        selected_log_pattern_dict = self.get_selected_log_pattern_features()
        language_service = self.config.data_dict['setting']['service_knowledge']['language_service']
        for language, service_list in language_service.items():
            for service in service_list:
                for i in selected_log_pattern_dict[language]:
                    selected_log_pattern_features.append(f'{service}; <log pattern {i}>')

        raw_data = self.load_raw_log('container')
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['container_log_features']
                    for feature_name in feature_df.keys():
                        if feature_name == 'timestamp':
                            continue
                        exact_feature_name = LogGenerator.extract_entity_feature_name(feature_name)
                        if exact_feature_name in selected_log_pattern_features:
                            if exact_feature_name not in data_dict.keys():
                                statistic_dict[exact_feature_name] = 0
                                data_dict[exact_feature_name] = []
                            data_dict[exact_feature_name].extend(raw_data[dataset_type][date][cloud_bed]['container_log_features'][feature_name].tolist())

        for feature_name in statistic_dict.keys():
            log_data = data_dict[feature_name]
            median = np.nanmedian(log_data)
            percentile_1 = np.nanpercentile(log_data, 1)
            percentile_99 = np.nanpercentile(log_data, 99)
            q1 = np.nanpercentile(log_data, 25)
            q3 = np.nanpercentile(log_data, 75)
            mean = np.nanmean(log_data)
            std = np.nanstd(log_data)
            valid_ratio = (np.count_nonzero(~np.isnan(log_data))) / len(list(log_data))

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

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/log'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/container_log_statistic.json', 'w') as f:
            json.dump(statistic_dict, f, indent=2)

    def z_score_container_log_data(self):
        raw_data = self.load_raw_log('container')
        selected_log_pattern_feature_dict = self.get_selected_log_pattern_features()

        file_dict = self.config.data_dict['file']
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/log/container_log_statistic.json', 'r') as f:
            statistic_dict = json.load(f)

        selected_feature_dict = dict()
        for dataset_type, dataset_detail_dict in file_dict.items():
            selected_feature_dict[dataset_type] = dict()
            for date in dataset_detail_dict['date']:
                selected_feature_dict[dataset_type][date] = dict()
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    selected_feature_dict[dataset_type][date][cloud_bed] = {
                        'container_log_features': dict(),
                        'raw_container_log_features': dict()
                    }
                    feature_df = raw_data[dataset_type][date][cloud_bed]['container_log_features']
                    selected_feature_dict[dataset_type][date][cloud_bed]['container_log_features']['timestamp'] = feature_df['timestamp']
                    selected_feature_dict[dataset_type][date][cloud_bed]['raw_container_log_features']['timestamp'] = feature_df['timestamp']
                    for pod in self.config.data_dict['setting']['metric']['pod_order']:
                        service_language = self.config.data_dict['setting']['service_knowledge']['service_language'][LogGenerator.rename_pod2service(pod)]
                        for language, selected_pattern_list in selected_log_pattern_feature_dict.items():
                            for i in selected_pattern_list:
                                if language == service_language:
                                    exact_feature_name = LogGenerator.extract_entity_feature_name(f'{pod}; <log pattern {i}>')
                                    raw_log_feature_data = raw_data[dataset_type][date][cloud_bed]['container_log_features'][f'{pod}; <log pattern {i}>']
                                    if statistic_dict[exact_feature_name]['std'] != 0:
                                        selected_feature_dict[dataset_type][date][cloud_bed]['container_log_features'][f'{pod}; {language}; <log pattern {i}>'] = (raw_log_feature_data - statistic_dict[exact_feature_name]['mean']) / statistic_dict[exact_feature_name]['std']
                                    else:
                                        selected_feature_dict[dataset_type][date][cloud_bed]['container_log_features'][f'{pod}; {language}; <log pattern {i}>'] = [0 for _ in range(feature_df.shape[0])]
                                    selected_feature_dict[dataset_type][date][cloud_bed]['raw_container_log_features'][f'{pod}; {language}; <log pattern {i}>'] = raw_log_feature_data
                                else:
                                    selected_feature_dict[dataset_type][date][cloud_bed]['container_log_features'][f'{pod}; {language}; <log pattern {i}>'] = [0 for _ in range(feature_df.shape[0])]
                                    selected_feature_dict[dataset_type][date][cloud_bed]['raw_container_log_features'][f'{pod}; {language}; <log pattern {i}>'] = [0 for _ in range(feature_df.shape[0])]
                    selected_feature_dict[dataset_type][date][cloud_bed]['container_log_features'] = pd.DataFrame(selected_feature_dict[dataset_type][date][cloud_bed]['container_log_features'])
                    selected_feature_dict[dataset_type][date][cloud_bed]['raw_container_log_features'] = pd.DataFrame(selected_feature_dict[dataset_type][date][cloud_bed]['raw_container_log_features'])
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/log'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/z_scored_container_log_features.pkl', 'wb') as f:
            pickle.dump(selected_feature_dict, f)

    def calculate_word_tf_idf(self):
        os.makedirs(f'{self.config.param_dict["temp_data_storage"]}/analysis/log')
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/log/istio_words.json') as f:
            raw_istio_words = json.load(f)['word_set']

        istio_words = [word for word in raw_istio_words if 'latency=' not in word and 'ttl=' not in word]

        result_dict = {
            'analysis': {
                'idf': dict(),
                'max_tf': dict(),
                'max_tf_idf': dict()
            },
            'raw_tf_idf': {
                'tf': dict(),
                'tf_idf': dict()
            }
        }
        temp_dict = dict()
        for word in istio_words:
            temp_dict[word] = []

        raw_data = self.load_raw_log('istio')
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['istio_log_features']
                    for word in istio_words:
                        temp_dict[word].extend((feature_df[word] / feature_df['total']).tolist())

        selected_word_list = []
        for word in istio_words:
            temp = np.array(temp_dict[word])
            idf = np.log(temp.shape[0] / (1 + np.count_nonzero(temp)))
            result_dict['analysis']['idf'][word] = idf
            result_dict['analysis']['max_tf'][word] = np.sort(temp)[-30:].mean()
            result_dict['analysis']['max_tf_idf'][word] = np.sort(temp)[-30:].mean() * idf
            result_dict['raw_tf_idf']['tf'][word] = temp.tolist()
            result_dict['raw_tf_idf']['tf_idf'][word] = (temp * idf).tolist()
            if result_dict['analysis']['max_tf_idf'][word] > 0.04:
                print(f'word: {word}; idf: {idf}; max_tf: {result_dict["analysis"]["max_tf"][word]}; '
                      f'max_tf_idf: {result_dict["analysis"]["max_tf_idf"][word]}')
                selected_word_list.append(word)

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/log'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/istio_log_tf_idf_statistic.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

        with open(f'{folder}/istio_selected_words.json', 'w') as f:
            json.dump(selected_word_list, f, indent=2)

    def calculate_istio_log_statistic(self):
        statistic_dict = dict()
        data_dict = dict()

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/log'
        with open(f'{folder}/istio_selected_words.json') as f:
            selected_words = json.load(f)

        raw_data = self.load_raw_log('istio')
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    for pod in self.config.data_dict['setting']['metric']['pod_order']:
                        for word in selected_words:
                            exact_feature_name = LogGenerator.extract_entity_feature_name(f'{pod}; {word}')
                            if exact_feature_name not in data_dict.keys():
                                statistic_dict[exact_feature_name] = 0
                                data_dict[exact_feature_name] = []
                            data_dict[exact_feature_name].extend(raw_data[dataset_type][date][cloud_bed]['istio_log_features'][f'{pod}; {word}'].tolist())

        for feature_name in statistic_dict.keys():
            log_data = data_dict[feature_name]
            median = np.nanmedian(log_data)
            percentile_1 = np.nanpercentile(log_data, 1)
            percentile_99 = np.nanpercentile(log_data, 99)
            q1 = np.nanpercentile(log_data, 25)
            q3 = np.nanpercentile(log_data, 75)
            mean = np.nanmean(log_data)
            std = np.nanstd(log_data)
            valid_ratio = (np.count_nonzero(~np.isnan(log_data))) / len(list(log_data))

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

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/log'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/istio_log_statistic.json', 'w') as f:
            json.dump(statistic_dict, f, indent=2)

    def z_score_istio_log_data(self):
        raw_data = self.load_raw_log('istio')

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/log'
        with open(f'{folder}/istio_selected_words.json') as f:
            selected_words = json.load(f)

        file_dict = self.config.data_dict['file']
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/log/istio_log_statistic.json', 'r') as f:
            statistic_dict = json.load(f)

        for dataset_type, dataset_detail_dict in file_dict.items():
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['istio_log_features']
                    selected_feature_dict, raw_selected_feature_dict = dict(), dict()
                    selected_feature_dict['timestamp'], raw_selected_feature_dict['timestamp'] = feature_df['timestamp'], feature_df['timestamp']
                    for pod in self.config.data_dict['setting']['metric']['pod_order']:
                        for word in selected_words:
                            exact_feature_name = LogGenerator.extract_entity_feature_name(f'{pod}; {word}')
                            raw_log_feature_data = feature_df[f'{pod}; {word}']
                            if statistic_dict[exact_feature_name]['std'] != 0:
                                selected_feature_dict[f'{pod}; {word}'] = (raw_log_feature_data - statistic_dict[exact_feature_name]['mean']) / statistic_dict[exact_feature_name]['std']
                            else:
                                selected_feature_dict[f'{pod}; {word}'] = [0 for _ in range(len(raw_log_feature_data))]
                            raw_selected_feature_dict[f'{pod}; {word}'] = raw_log_feature_data
                    raw_data[dataset_type][date][cloud_bed]['istio_log_features'] = pd.DataFrame(selected_feature_dict)
                    raw_data[dataset_type][date][cloud_bed]['raw_istio_log_features'] = pd.DataFrame(raw_selected_feature_dict)

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/log'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/z_scored_istio_features.pkl', 'wb') as f:
            pickle.dump(raw_data, f)

    def slice_log_features(self):
        result_dict = dict()

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/log'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/z_scored_container_log_features.pkl', 'rb') as f:
            log_pattern_dict = pickle.load(f)
        with open(f'{folder}/z_scored_istio_features.pkl', 'rb') as f:
            istio_dict = pickle.load(f)
        container_list = self.config.data_dict['setting']['metric']['pod_order']
        file_dict = self.config.data_dict['file']

        log_pattern_features, istio_features = None, None
        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dict[dataset_type] = dict()
            for date in dataset_detail_dict['date']:
                result_dict[dataset_type][date] = dict()
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    result_dict[dataset_type][date][cloud_bed] = {
                        'log_features': dict(),
                        'raw_log_features': dict(),
                    }
                    container_log_df = log_pattern_dict[dataset_type][date][cloud_bed]['container_log_features']
                    istio_df = istio_dict[dataset_type][date][cloud_bed]['istio_log_features']
                    raw_container_log_df = log_pattern_dict[dataset_type][date][cloud_bed]['raw_container_log_features']
                    raw_istio_df = istio_dict[dataset_type][date][cloud_bed]['raw_istio_log_features']
                    for i in range(len(container_list)):
                        temp_dict, raw_temp_dict = {
                            'timestamp': container_log_df['timestamp']
                        }, {
                            'timestamp': raw_container_log_df['timestamp']
                        }

                        if log_pattern_features is None:
                            log_pattern_features = container_log_df.columns[container_log_df.columns.str.contains(f'{container_list[i]}; ')].map(lambda x: str(x).replace(f'{container_list[i]}; ', ""))
                        for feature_name in log_pattern_features:
                            temp_dict[feature_name] = container_log_df[f'{container_list[i]}; {feature_name}']
                            raw_temp_dict[feature_name] = raw_container_log_df[f'{container_list[i]}; {feature_name}']

                        if istio_features is None:
                            istio_features = istio_df.columns[istio_df.columns.str.contains(f'{container_list[i]}; ')].map(lambda x: str(x).replace(f'{container_list[i]}; ', ""))
                        for feature_name in istio_features:
                            temp_dict[feature_name] = istio_df[f'{container_list[i]}; {feature_name}']
                            raw_temp_dict[feature_name] = raw_istio_df[f'{container_list[i]}; {feature_name}']
                        result_dict[dataset_type][date][cloud_bed]['log_features'][container_list[i]] = pd.DataFrame(temp_dict)
                        result_dict[dataset_type][date][cloud_bed]['raw_log_features'][container_list[i]] = pd.DataFrame(raw_temp_dict)

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/log'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/all_log.pkl', 'wb') as f:
            pickle.dump(result_dict, f)

    def get_all_log(self) -> dict:
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/log'
        with open(f'{folder}/all_log.pkl', 'rb') as f:
            log = pickle.load(f)
        return log


if __name__ == '__main__':
    log_generator = LogGenerator()
    log_generator.calculate_log_pattern_tf_idf()
    log_generator.calculate_container_log_pattern_statistic()
    log_generator.z_score_container_log_data()
    log_generator.calculate_word_tf_idf()
    log_generator.calculate_istio_log_statistic()
    log_generator.z_score_istio_log_data()
    log_generator.slice_log_features()
