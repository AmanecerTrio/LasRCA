import os

import numpy as np
import pickle

from data_filter.CCF_AIOps_challenge_2022.base.base_class import BaseClass


class TopologyDao(BaseClass):
    def __init__(self):
        super().__init__()

    def add_service_pod_topology(self, topology_edge_index: list):
        for service in self.config.data_dict['setting']['metric']['service_order']:
            service_index = self.all_entity_list.index(service)
            for pod_item in ['-0', '-1', '-2', '2-0']:
                topology_edge_index[0].append(self.all_entity_list.index(f'{service}{pod_item}'))
                topology_edge_index[1].append(service_index)
                topology_edge_index[1].append(self.all_entity_list.index(f'{service}{pod_item}'))
                topology_edge_index[0].append(service_index)
        return topology_edge_index

    def add_service_service_topology(self, topology_edge_index: list):
        call_relation_list = [
            'frontend -> adservice',
            'frontend -> cartservice',
            'frontend -> checkoutservice',
            'frontend -> currencyservice',
            'frontend -> recommendationservice',
            'frontend -> productcatalogservice',
            'frontend -> shippingservice',
            'checkoutservice -> cartservice',
            'checkoutservice -> currencyservice',
            'checkoutservice -> emailservice',
            'checkoutservice -> paymentservice',
            'checkoutservice -> productcatalogservice',
            'checkoutservice -> shippingservice',
            'recommendationservice -> productcatalogservice',
        ]
        for call_relation in call_relation_list:
            caller, callee = call_relation.split(' -> ')[0], call_relation.split(' -> ')[1]
            topology_edge_index[0].append(self.all_entity_list.index(caller))
            topology_edge_index[1].append(self.all_entity_list.index(callee))
            topology_edge_index[0].append(self.all_entity_list.index(callee))
            topology_edge_index[1].append(self.all_entity_list.index(caller))
        return topology_edge_index

    def add_self_loops_topology(self, topology_edge_index: list):
        for i in range(len(self.all_entity_list)):
            topology_edge_index[0].append(i)
            topology_edge_index[1].append(i)
        return topology_edge_index

    def generate_topology_edge_index(self):
        file_dict = self.config.data_dict['file']
        edge_index_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'
        edge_index_dict = dict()

        for dataset_type, dataset_detail_dict in file_dict.items():
            edge_index_dict[dataset_type] = dict()
            edge_index_dataset_type_path = f'{edge_index_base_path}/{dataset_type}'
            for date in dataset_detail_dict['date']:
                edge_index_dict[dataset_type][date] = dict()
                edge_index_date_path = f'{edge_index_dataset_type_path}/{date}'
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    topology_edge_index = [[], []]

                    topology_edge_index = self.add_service_pod_topology(topology_edge_index)
                    topology_edge_index = self.add_service_service_topology(topology_edge_index)
                    topology_edge_index = self.add_self_loops_topology(topology_edge_index)

                    result_base_path = f'{edge_index_date_path}/{cloud_bed}/resource_entity'
                    os.makedirs(result_base_path, exist_ok=True)

                    with open(f'{result_base_path}/edge_index.pkl', 'wb') as f:
                        pickle.dump(topology_edge_index, f)

    def load_topology_edge_index(self):
        result_dict = dict()
        file_dict = self.config.data_dict['file']
        topology_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'
        for dataset_type, dataset_detail_dict in file_dict.items():
            topology_dataset_type_path = f'{topology_base_path}/{dataset_type}'
            for date in dataset_detail_dict['date']:
                topology_date_path = f'{topology_dataset_type_path}/{date}'
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    with open(f'{topology_date_path}/{cloud_bed}/resource_entity/edge_index.pkl', 'rb') as f:
                        result_dict[f'{date}/{cloud_bed}'] = pickle.load(f)
        return result_dict


if __name__ == '__main__':
    topology_dao = TopologyDao()
    topology_dao.generate_topology_edge_index()
    topology_dao.load_topology_edge_index()
