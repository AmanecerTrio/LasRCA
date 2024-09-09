from abc import ABC
from data_filter.CCF_AIOps_challenge_2022.config.dataset_config import DataConfig
from shared_util.common import *


class BaseClass(ABC):
    def __init__(self):
        self.config = DataConfig()
        self.all_entity_list = []
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['node_order'])
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['service_order'])
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['pod_order'])

    @staticmethod
    def rename_pod2service(pod_name):
        return pod_name.replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '')

    @staticmethod
    def rename_service2pod(service_name):
        return [f'{service_name}-0', f'{service_name}-1', f'{service_name}-2', f'{service_name}2-0']
