import os
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator
import pickle


class EntEdgeIndexGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def extract_topology(self):
        topology_dict = self.topology_dao.load_topology_edge_index()
        return topology_dict

    def save_ent_edge_index(self):
        topology_dict = self.extract_topology()

        relation_dict = {
            'normal': topology_dict,
            'train_valid': topology_dict,
            'test': topology_dict
        }
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/ent_edge_index'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/ent_edge_index.pkl', 'wb') as f:
            pickle.dump(relation_dict, f)

    def get_ent_edge_index(self):
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/ent_edge_index'
        with open(f'{folder}/ent_edge_index.pkl', 'rb') as f:
            ent_edge_index = pickle.load(f)
            return ent_edge_index


if __name__ == '__main__':
    ent_edge_index_generator = EntEdgeIndexGenerator()
    ent_edge_index_generator.save_ent_edge_index()
