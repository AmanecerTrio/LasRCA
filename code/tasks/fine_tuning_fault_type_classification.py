import sys

sys.path.append('/workspace/project/working/2024/LasRCA/code')

from abc import ABC
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from shared_util.common import *
from shared_util.seed import *
from tasks.base import *
import argparse
from shared_util.common import *
from shared_util.seed import *
from tasks.models.GAT_net import GraphNet
from tasks.util import label_to_multi_class_format, rearrange_y
from tasks.eval import evaluate_ftc
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve


class DLDataset(Dataset):
    """
    生成PyTorch加载数据需要的Dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        item = dict()
        for key in self.data.keys():
            if key == 'embeddings':
                item[key] = torch.FloatTensor(np.array(self.data[key][idx]).astype(np.float64))
            if key == 'ent_edge_index':
                item[key] = torch.LongTensor(self.data[key][idx])
            if key == 'y':
                item[key] = torch.FloatTensor(np.array(self.data[key][idx]).astype(np.float64))
        return item


class BaseClass(ABC):
    def __init__(self, data_config, model_config):
        self.base_path = "/workspace/project/working/2024/LasRCA/temp_data"
        self.raw_data = pkl_load(f'{self.base_path}/{data_config["dataset_path"]}')
        self.embeddings = pkl_load(f'{self.base_path}/{data_config["embedding_path"]}') if data_config['embedding_path'] is not None else None
        self.ground_truth = pkl_load(f'{self.base_path}/{data_config["ground_truth_path"]}')
        self.data_config = data_config
        self.model_config = model_config
        self.dataset = {
            'meta_data': ...,
            'data': dict()
        }
        self.prepare_dataset()
        self.model = GraphNet(self.model_config, self.dataset['meta_data']).to(self.model_config['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])

        self.ent_fault_type_weight = dict()
        for ent_type in self.dataset['meta_data']['ent_types']:
            ent_fault_index_pair = self.dataset['meta_data']['ent_fault_type_index'][ent_type]
            ent_index_pair = self.dataset['meta_data']['ent_type_index'][ent_type]
            temp = np.array(self.dataset['data']['train']['y'])[:, ent_index_pair[0]:ent_index_pair[1]]
            self.ent_fault_type_weight[ent_type] = [np.sum(temp == 0) / np.sum(temp == i) for i in range(ent_fault_index_pair[0] + 1, ent_fault_index_pair[1] + 1)]
            self.ent_fault_type_weight[ent_type] = torch.FloatTensor(self.ent_fault_type_weight[ent_type]).to( self.model_config['device'])
        self.criterion = {ent_type: torch.nn.BCEWithLogitsLoss(pos_weight=self.ent_fault_type_weight[ent_type]) for ent_type in self.dataset['meta_data']['ent_types']}
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=32, eta_min=0)
        self.fc_result = {ent_type: dict() for ent_type in self.dataset['meta_data']['ent_types']}

    def get_dataset_file_name(self):
        suffix_str = f'train_size_{self.data_config["each_train_size"] if "few_shot" in self.data_config["sample_strategy"] else self.data_config["train_size"]}'
        if self.data_config["sample_strategy"] != "narrow_few_shot":
            suffix_str += f'_valid_size_{self.data_config["each_valid_size"] if self.data_config["sample_strategy"] == "broad_few_shot" else self.data_config["valid_size"]}'
        file_name = f'fine_tuning_{self.data_config["sample_strategy"]}_{suffix_str}.pkl'
        save_file_path = f'{self.base_path}/{self.data_config["dataset"]}/dataset'
        return save_file_path, file_name

    def prepare_dataset(self, overwrite=False):
        save_file_path, file_name = self.get_dataset_file_name()
        if os.path.exists(f'{save_file_path}/{file_name}') and not overwrite:
            self.dataset = pkl_load(f'{save_file_path}/{file_name}')
            return
        os.makedirs(save_file_path, exist_ok=True)

        if 'A' in self.data_config['dataset']:
            self.dataset['meta_data'] = self.raw_data['meta_data']
            for dataset_type in ['train_valid', 'test']:
                self.dataset['data'][dataset_type] = {
                    'embeddings': [],
                    'data': [],
                    'ent_edge_index': [],
                    'raw_data': [],
                    'y': [],
                    'y_marker': []
                }
                for i in range(len(self.ground_truth['time_interval'][dataset_type])):
                    for key in self.dataset['data'][dataset_type].keys():
                        if key != 'y_marker' and key != 'ent_edge_index':
                            self.dataset['data'][dataset_type][key].append([])

                    self.dataset['data'][dataset_type]['y_marker'].append(f'{self.ground_truth["entity_type"][dataset_type][i]}/{self.ground_truth["root_cause_type"][dataset_type][i]}')

                    date = self.ground_truth['time_interval'][dataset_type][i][0]
                    cloud_bed = self.ground_truth['time_interval'][dataset_type][i][1]
                    start_timestamp = self.ground_truth['time_interval'][dataset_type][i][2]
                    end_timestamp = self.ground_truth['time_interval'][dataset_type][i][3]

                    total_timestamps = self.raw_data['data'][dataset_type][f'{date}/{cloud_bed}']['node']['node-1']['timestamp'].tolist()
                    start_index, end_index = total_timestamps.index(start_timestamp), total_timestamps.index(end_timestamp)

                    self.dataset['data'][dataset_type]['ent_edge_index'].append(self.dataset['meta_data']['ent_edge_index'][dataset_type][f'{date}/{cloud_bed}'])

                    # embeddings, preprocessed data, raw data.
                    for ent_type in self.raw_data['meta_data']['ent_types']:
                        ent_type_index_pair = self.raw_data['meta_data']['ent_type_index'][ent_type]
                        for ent_name in self.raw_data['meta_data']['ent_names'][ent_type_index_pair[0]:ent_type_index_pair[1]]:
                            self.dataset['data'][dataset_type]['embeddings'][-1].append(self.embeddings[dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name][start_index:end_index, :])
                            data_df = self.raw_data['data'][dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name]
                            self.dataset['data'][dataset_type]['data'][-1].append(data_df.iloc[start_index:end_index, data_df.columns != "timestamp"])
                            raw_data_df = self.raw_data['raw_data'][dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name]
                            self.dataset['data'][dataset_type]['raw_data'][-1].append(raw_data_df.iloc[start_index:end_index, raw_data_df.columns != "timestamp"])
                self.dataset['data'][dataset_type]['y'] = self.ground_truth['y'][dataset_type]

        index_dict = {
            'train': [],
            'valid': []
        }
        if 'few_shot' in self.data_config['sample_strategy']:
            y_marker_set = set(self.dataset['data']['train_valid']['y_marker'])
            for y_marker in y_marker_set:
                sample_list = [index for (index, value) in enumerate(self.dataset['data']['train_valid']['y_marker']) if
                               value == y_marker]
                if self.data_config["sample_strategy"] == "broad_few_shot":
                    total_size = self.data_config['each_train_size'] + self.data_config['each_valid_size']
                    if len(sample_list) < total_size:
                        fill_list = np.random.choice(sample_list, total_size - len(sample_list)).tolist()
                        sample_list.extend(fill_list)
                    chosen_list = np.random.choice(sample_list, total_size, replace=False).tolist()
                    index_dict['train'].extend(chosen_list[:self.data_config['each_train_size']])
                    index_dict['valid'].extend(chosen_list[self.data_config['each_train_size']:total_size])
                elif self.data_config["sample_strategy"] == "narrow_few_shot":
                    if len(sample_list) < self.data_config['each_train_size']:
                        fill_list = np.random.choice(sample_list,
                                                     self.data_config['each_train_size'] - len(sample_list)).tolist()
                        sample_list.extend(fill_list)
                    chosen_list = np.random.choice(sample_list, self.data_config['each_train_size'],
                                                   replace=False).tolist()
                    index_dict['train'].extend(chosen_list)
                    index_dict['valid'].extend(chosen_list)

            """
            NOTE: FROM SELECTED EXAMPLES
            """
            example_dir = "/workspace/project/working/2024/LasRCA/temp_data/A/example_selection"
            with open(f'{example_dir}/example_{self.data_config["each_train_size"]}.json', 'r') as f:
                example_dict = json.load(f)
            index_dict['train'] = example_dict['final_batch_indices']
            index_dict['valid'] = example_dict['final_batch_indices']

        elif self.data_config['sample_strategy'] == 'random':
            sample_list = list(range(len(self.dataset['data']['train_valid']['y_marker'])))
            index_dict['train'], index_dict['valid'] = train_test_split(sample_list, train_size=self.data_config['train_size'], random_state=rca_seed)
        for dataset_type, index_list in index_dict.items():
            self.dataset['data'][dataset_type] = dict()
            for key, value in self.dataset['data']['train_valid'].items():
                self.dataset['data'][dataset_type][key] = [value[i] for i in index_list]
        pkl_save(f'{save_file_path}/{file_name}', self.dataset)


class FineTuningFaultTypeClassification(BaseClass):
    def __init__(self, data_config, model_config):
        super().__init__(data_config, model_config)
        self.prepare_dataset()

    def evaluate_las_fault_classification(self, data_loader, dataset_type, threshold=0.5):
        self.model.eval()
        y_pred, y_true = dict(), dict()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader[dataset_type]):
                y = rearrange_y(self.dataset['meta_data'], batch_data['y'], self.model_config['device'])
                out = self.model(batch_data)
                for ent_type in self.dataset['meta_data']['ent_types']:
                    if ent_type not in y_pred.keys():
                        y_pred[ent_type], y_true[ent_type] = [], []
                    y_pred[ent_type].extend(torch.sigmoid(out[ent_type]).cpu().detach().numpy())
                    y_true[ent_type].extend(y[ent_type].cpu().detach().numpy())
        for ent_type in self.dataset['meta_data']['ent_types']:
            y_pred[ent_type], y_true[ent_type] = np.array(y_pred[ent_type]), np.array(y_true[ent_type])
            y_pred[ent_type] = (np.array(y_pred[ent_type]) > threshold).tolist()

        evaluate_result = evaluate_ftc(y_pred, y_true, dataset_type, self.dataset['meta_data']['ent_types'])
        for ent_type, fc_result in evaluate_result.items():
            for em in ['precision', 'recall', 'f1']:
                for prefix in ['micro_', 'macro_']:
                    full_em = f'{prefix}{em}_score'
                    if full_em not in self.fc_result[ent_type].keys():
                        self.fc_result[ent_type][full_em] = []
                    self.fc_result[ent_type][full_em].append(format(fc_result[full_em], '.6f'))
            for em in ['precision', 'recall', 'f1']:
                full_em = f'{em}_score'
                if full_em not in self.fc_result[ent_type].keys():
                    self.fc_result[ent_type][full_em] = []
                self.fc_result[ent_type][full_em].append(str(fc_result[full_em]))

    def train_based_on_representation(self):
        data_loader = dict()
        for dataset_type in self.dataset['data'].keys():
            self.dataset['data'][dataset_type]['y'] = label_to_multi_class_format(self.dataset['data'][dataset_type]['y'], num_of_fault_types=len(self.dataset['meta_data']['fault_type_list']))
            if dataset_type == 'train':
                data_loader[dataset_type] = DataLoader(DLDataset(self.dataset['data'][dataset_type]), batch_size=self.model_config['batch_size'], shuffle=True)
            else:
                data_loader[dataset_type] = DataLoader(DLDataset(self.dataset['data'][dataset_type]), batch_size=self.model_config['batch_size'], shuffle=False)

        for epoch in range(self.model_config['epochs']):
            self.model.train()
            train_loss = 0
            for batch_id, batch_data in enumerate(data_loader['train']):
                self.optimizer.zero_grad()
                out = self.model(batch_data)
                y = rearrange_y(self.dataset['meta_data'], batch_data['y'], self.model_config['device'])
                loss = 0
                for ent_type in self.dataset['meta_data']['ent_types']:
                    loss += self.criterion[ent_type](out[ent_type], y[ent_type])
                train_loss += batch_data['y'].shape[0] * loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            logger.info(f'[{epoch}/{self.model_config["epochs"]}] | train_loss: {train_loss:.5f}')

        self.evaluate_las_fault_classification(data_loader, 'test')
        torch.save(self.model.state_dict(), f'{self.data_config["model_dir"]}/model_seed_{self.model_config["seed"]}.pkl')

    def final_test(self):
        data_loader = dict()
        for dataset_type in self.dataset['data'].keys():
            self.dataset['data'][dataset_type]['y'] = label_to_multi_class_format(self.dataset['data'][dataset_type]['y'], num_of_fault_types=len(self.dataset['meta_data']['fault_type_list']))
            if dataset_type == 'train':
                data_loader[dataset_type] = DataLoader(DLDataset(self.dataset['data'][dataset_type]), batch_size=self.model_config['batch_size'], shuffle=True)
            else:
                data_loader[dataset_type] = DataLoader(DLDataset(self.dataset['data'][dataset_type]), batch_size=self.model_config['batch_size'], shuffle=False)
        self.model.load_state_dict(torch.load(f'{self.data_config["model_dir"]}/model_seed_{self.model_config["seed"]}.pkl'))
        self.evaluate_las_fault_classification(data_loader, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    window_size = 9
    parser.add_argument('--dataset', default='A/metric_trace_log')
    parser.add_argument('--dataset_path', default='2022_CCF_AIOps_challenge/dataset/final/metric_trace_log.pkl')
    parser.add_argument('--embedding_path', default='A/embeddings/embedding_600.pkl')
    parser.add_argument('--ground_truth_path', default=f'2022_CCF_AIOps_challenge/dataset/time_interval_and_label/time_interval_window_size_{window_size}.pkl')

    parser.add_argument('--sample_strategy', default='narrow_few_shot', help='random or narrow_few_shot or broad_few_shot')
    parser.add_argument('--train_size', type=int, default=200)
    parser.add_argument('--valid_size', type=int, default=100)
    parser.add_argument('--each_train_size', type=int, default=1)
    parser.add_argument('--each_valid_size', type=int, default=1)
    parser.add_argument('--window_size', default=window_size)

    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--seed', type=int, default=405, help='Random seed for deterministic')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size (defaults to 8)')
    parser.add_argument('--epochs', type=int, default=500, help='The number of epochs')
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)

    parser.add_argument("--in_dim", default=320, type=int)
    parser.add_argument("--GAT_out_channels", default=128, type=int)
    parser.add_argument("--GAT_heads", default=2, type=int)
    parser.add_argument("--GAT_dropout", default=0.1, type=float)

    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    args = parser.parse_args()

    device = init_dl_program(args.gpu, max_threads=args.max_threads)

    args = vars(args)

    seed_everything(seed=args["seed"])
    args['device'] = device

    model_dir = f'/workspace/project/working/2024/LasRCA/model/fine-tune/{args["dataset"]}'
    os.makedirs(model_dir, exist_ok=True)
    test_data_config = {
        'dataset': f'{args["dataset"]}',
        'dataset_path': f'{args["dataset_path"]}',
        'embedding_path': f'{args["embedding_path"]}',
        'ground_truth_path': f'{args["ground_truth_path"]}',
        'model_dir': f'{model_dir}',
        'sample_strategy': args['sample_strategy'],
        'train_size': args['train_size'],
        'valid_size': args['valid_size'],
        'each_train_size': args['each_train_size'],
        'each_valid_size': args['each_valid_size'],
        'window_size': args['window_size']
    }
    # fault_type_classification = FineTuningFaultTypeClassification(test_data_config, args)
    # fault_type_classification.prepare_dataset()
    # fault_type_classification.train_based_on_representation()

    final_dict = dict()
    for seed in [405, 406, 407, 408, 409]:
        args['seed'] = seed
        fault_type_classification = FineTuningFaultTypeClassification(test_data_config, args)
        fault_type_classification.final_test()
        for result in [fault_type_classification.fc_result]:
            for ent_type in result.keys():
                if ent_type not in final_dict.keys():
                    final_dict[ent_type] = dict()
                for em in result[ent_type].keys():
                    if em == 'precision_score' or em == 'recall_score' or em == 'f1_score':
                        continue
                    if em not in final_dict[ent_type].keys():
                        final_dict[ent_type][em] = []
                    temp = result[ent_type][em]
                    while isinstance(temp, list):
                        temp = temp[0]
                    temp = float(temp)
                    final_dict[ent_type][em].append(temp)
    for ent_type in final_dict.keys():
        for em in final_dict[ent_type].keys():
            print(f'{ent_type} | {em} | mean: {np.mean(final_dict[ent_type][em])}; std: {np.std(final_dict[ent_type][em])}')
