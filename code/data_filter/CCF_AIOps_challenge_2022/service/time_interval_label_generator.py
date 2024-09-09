import os

import numpy as np
from tqdm import tqdm

from shared_util.common import *
from shared_util.time_handler import TimeHandler
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator


class TimeIntervalLabelGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def get_ground_truth(self, ground_truth_dict, index):
        """
        重组每个ground truth.
        :param ground_truth_dict: 某个date下的cloud_bed下的ground_truth记录.
        :param index: 索引.
        :return: 如下所示的dict.
        """
        return {
            'timestamp': ground_truth_dict['timestamp'][index],
            'level': ground_truth_dict['level'][index],
            'cmdb_id': ground_truth_dict['cmdb_id'][index],
            'fault_type': self.fault_type_list.index(ground_truth_dict['failure_type'][index]) + 1
        }

    @staticmethod
    def get_date_timestamp_list(date_str: str) -> list:
        """
        给定一个日期, 获取当天每分钟的时间戳, 便于划分时间窗口.
        :param date_str: %Y-%m-%d格式的日期字符串.
        :return: list, 包含了当天每分钟对应时间戳的数组.
        """
        date_start_timestamp = TimeHandler.datetime_to_timestamp(date_str + ' 00:00:00')
        return list(range(date_start_timestamp, date_start_timestamp + 24 * 60 * 60, 60))

    def get_ground_truth_label(self, ground_truth):
        """
        获取ground_truth对应的label表示.
        :param ground_truth: self.get_ground_truth中的返回结构.
        :return: np.array, 长度为实体数量, 每一位代表实体的标签, 0代表无故障, 1-15代表某一种故障, 具体对应详见data_filter/CCF_AIOps_challenge_2022/base/base_generator.py.
        """
        label = np.zeros(len(self.all_entity_list))
        label[self.all_entity_list.index(ground_truth['cmdb_id'])] = ground_truth['fault_type']
        if ground_truth['level'] == 'service':
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-0')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-1')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-2')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}2-0')] = ground_truth['fault_type']
        return label

    def slice_normal_timestamp(self, window_size):
        """
        根据正常天数, 通过滑动窗口截取正常数据. (因为前后20分钟可能存在系统波动, 因此去除前后20min的数据.)
        :param window_size: 窗口大小.
        :return: list, 每一项为(date, interval_start_time, interval_end_time).
        """
        interval_list = []
        date, cloud_bed_list = '2022-03-19', ['cloudbed-1', 'cloudbed-2', 'cloudbed-3']
        start_timestamp = TimeHandler.datetime_to_timestamp(date + ' 00:00:00')
        for i in range(20, 1420 - window_size):
            s_ts = start_timestamp + i * 60
            e_ts = s_ts + window_size * 60
            for cloud_bed in cloud_bed_list:
                interval_list.append((date, cloud_bed, s_ts, e_ts))
        return interval_list

    def slice_ground_truth_timestamp(self, date, cloud_bed, ground_truth_timestamp, window_size, sliding_ratio):
        """
        划分出故障数据的time interval.
        :param date: %Y-%m-%d格式的日期字符串.
        :param cloud_bed: 属于哪一个cloud_bed.
        :param ground_truth_timestamp: ground_truth的具体时间.
        :param window_size: 窗口大小.
        :param sliding_ratio: 滑动窗口起始位置与故障发生时间差值 / 滑动窗口长度, 即每次故障对应的滑动窗口数量.
        :return: list, 每一项为(date, interval_start_time, interval_end_time).
        """
        interval_list = []
        start_timestamp = TimeHandler.datetime_to_timestamp(date + ' 00:00:00')
        c_ts = ground_truth_timestamp - (ground_truth_timestamp - start_timestamp) % 60
        s_ts = c_ts - int(window_size * sliding_ratio) * 60
        e_ts = s_ts + window_size * 60
        interval_list.append((date, cloud_bed, s_ts, e_ts))
        return interval_list

    def generate_time_interval_label(self):
        """
        生成指定窗口长度的time_interval - label数据并存储到pkl中.
        样本数 = 故障样本 * window_size / 2.
        """
        window_size_bar = tqdm(self.window_size_list)
        for window_size in window_size_bar:
            faulty_time_interval, faulty_y = {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}
            faulty_entity_type, faulty_template, faulty_cmdb_id, faulty_root_cause_type = {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}

            for dataset_type in ['train_valid', 'test']:
                # 记录各个cloud bed上发生的故障信息, key: cloud_bed, value: ground_truth发生的时间戳list.
                train_ground_truth_timestamp_dict = dict()

                # 生成包含故障的天数的ground_truth对应的time_interval与label.
                for date, cloud_dict in self.ground_truth_dao.get_ground_truth(dataset_type).items():
                    for cloud_bed in cloud_dict.keys():
                        train_ground_truth_timestamp_dict[f'{date}/{cloud_bed}'] = []

                        for i in range(len(cloud_dict[cloud_bed]['timestamp'])):
                            ground_truth = self.get_ground_truth(cloud_dict[cloud_bed], i)
                            train_ground_truth_timestamp_dict[f'{date}/{cloud_bed}'].append(ground_truth['timestamp'])
                            temp_time_interval_list = self.slice_ground_truth_timestamp(date, cloud_bed, ground_truth['timestamp'], window_size, 0.5)
                            faulty_time_interval[dataset_type].extend(temp_time_interval_list)
                            faulty_y[dataset_type].extend([self.get_ground_truth_label(ground_truth) for i in range(len(temp_time_interval_list))])
                            faulty_entity_type[dataset_type].append(ground_truth['level'])
                            faulty_template[dataset_type].append(ground_truth['cmdb_id'].replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '').replace('-3', '').replace('-4', '').replace('-5', '').replace('-6', ''))
                            faulty_cmdb_id[dataset_type].append(ground_truth['cmdb_id'])
                            faulty_root_cause_type[dataset_type].append(self.fault_type_list[ground_truth['fault_type'] - 1])

            folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/time_interval_and_label'
            os.makedirs(folder, exist_ok=True)
            with open(f'{folder}/time_interval_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'time_interval': {
                        'normal': self.slice_normal_timestamp(window_size),
                        'train_valid': faulty_time_interval['train_valid'],
                        'test': faulty_time_interval['test']
                    },
                    'y': {
                        'normal': [np.zeros(len(self.all_entity_list)) for _ in self.slice_normal_timestamp(window_size)],
                        'train_valid': faulty_y['train_valid'],
                        'test': faulty_y['test']
                    },
                    'entity_type': {
                        'train_valid': faulty_entity_type['train_valid'],
                        'test': faulty_entity_type['test']
                    },
                    'template': {
                        'train_valid': faulty_template['train_valid'],
                        'test': faulty_template['test']
                    },
                    'cmdb_id': {
                        'train_valid': faulty_cmdb_id['train_valid'],
                        'test': faulty_cmdb_id['test']
                    },
                    'root_cause_type': {
                        'train_valid': faulty_root_cause_type['train_valid'],
                        'test': faulty_root_cause_type['test']
                    }
                }, f)
            window_size_bar.set_description("Time interval and label generating".format(window_size))

    def get_time_interval_label(self, window_size) -> dict:
        """
        获取不同窗口下的time_interval - label数据.
        :param window_size: 窗口长度.
        :return: 写在pickle中的数据格式.
        """
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/time_interval_and_label'
        with open(f'{folder}/time_interval_window_size_{window_size}.pkl', 'rb') as f:
            time_interval_label = pickle.load(f)
            return time_interval_label


if __name__ == '__main__':
    time_interval_label_generator = TimeIntervalLabelGenerator()
    time_interval_label_generator.generate_time_interval_label()
    time_interval_label_generator.get_time_interval_label(9)
