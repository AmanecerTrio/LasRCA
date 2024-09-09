import copy
from abc import ABC
from sklearn.model_selection import train_test_split
from shared_util.seed import *
from tasks.eval import evaluate_ftc
from tasks.models.o11y_representation_learning import RepresentationLearning
from tasks.models.feature_integration import FeatureIntegration
from tasks.models.feature_fusion import FeatureFusion
from tasks.models.fault_classifier import FaultClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR
from tasks.util import *


class BaseClass(ABC):
    def __init__(self, data_config, model_config):
        self.base_path = "/workspace/project/working/2024/LasRCA/temp_data"
        self.raw_data = pkl_load(f'{self.base_path}/{data_config["dataset_path"]}')
        self.ground_truth = pkl_load(f'{self.base_path}/{data_config["ground_truth_path"]}')
        self.data_config = data_config
        self.model_config = model_config
        self.dataset = {
            'meta_data': ...,
            'data': dict()
        }
        self.model = ...
        self.prepare_dataset()
        o11y_representation_learning = RepresentationLearning(model_config, self.dataset['meta_data'])
        re_feature_integration = FeatureIntegration(model_config, self.dataset['meta_data'])
        re_feature_fusion = FeatureFusion(model_config, self.dataset['meta_data'])
        re_fault_classifier = FaultClassifier(model_config, self.dataset['meta_data'])
        self.model = torch.nn.Sequential(o11y_representation_learning, re_feature_integration, re_feature_fusion, re_fault_classifier).to(model_config['device'])
        if self.model_config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])
        elif self.model_config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'], momentum=0.9, nesterov=True)

        self.ent_fault_type_weight = dict()
        for ent_type in self.dataset['meta_data']['ent_types']:
            ent_fault_index_pair = self.dataset['meta_data']['ent_fault_type_index'][ent_type]
            ent_index_pair = self.dataset['meta_data']['ent_type_index'][ent_type]
            temp = np.array(self.dataset['data']['train']['y'])[:, ent_index_pair[0]:ent_index_pair[1]]

            iter_labeled_temp = np.array(self.dataset['data']['unlabeled']['y'])[:, ent_index_pair[0]:ent_index_pair[1]]
            iter_labeled_temp_mask = np.array(self.dataset['data']['unlabeled']['y_mask'])[:, ent_index_pair[0]:ent_index_pair[1]]

            self.ent_fault_type_weight[ent_type] = [(np.sum(temp == 0) + (np.sum((iter_labeled_temp + iter_labeled_temp_mask) == 1))) / (np.sum(temp == i) + np.sum(iter_labeled_temp == i)) for i in range(ent_fault_index_pair[0] + 1, ent_fault_index_pair[1] + 1)]
            self.ent_fault_type_weight[ent_type] = torch.FloatTensor(self.ent_fault_type_weight[ent_type]).to(self.model_config['device'])
        self.criterion = {ent_type: torch.nn.BCEWithLogitsLoss(pos_weight=self.ent_fault_type_weight[ent_type]) for ent_type in self.dataset['meta_data']['ent_types']}
        self.u_criterion = {ent_type: torch.nn.BCEWithLogitsLoss(pos_weight=self.ent_fault_type_weight[ent_type], reduction='none') for ent_type in self.dataset['meta_data']['ent_types']}
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=32, eta_min=0)
        self.fl_result, self.fc_result = dict(), {ent_type: dict() for ent_type in self.dataset['meta_data']['ent_types']}

    def get_dataset_file_name(self, k=0):
        suffix_str = f'train_size_{self.data_config["each_train_size"] if "few_shot" in self.data_config["sample_strategy"] else self.data_config["train_size"]}'
        if self.data_config["sample_strategy"] != "narrow_few_shot":
            suffix_str += f'_valid_size_{self.data_config["each_valid_size"] if self.data_config["sample_strategy"] == "broad_few_shot" else self.data_config["valid_size"]}'
        if k != 0:
            suffix_str += f'_k_{k}_{self.model_config["labeling_strategy"]}'

        file_name = f'{self.data_config["sample_strategy"]}_{suffix_str}.pkl'
        save_file_path = f'{self.base_path}/{self.data_config["dataset"]}/dataset'
        return save_file_path, file_name

    def prepare_dataset(self, overwrite=False):
        save_file_path, file_name = self.get_dataset_file_name(self.model_config['k'])
        if os.path.exists(f'{save_file_path}/{file_name}') and not overwrite:
            self.dataset = pkl_load(f'{save_file_path}/{file_name}')
            return
        os.makedirs(save_file_path, exist_ok=True)

        if 'A' in self.data_config['dataset']:
            self.dataset['meta_data'] = self.raw_data['meta_data']
            for dataset_type in ['train_valid', 'test']:
                self.dataset['data'][dataset_type] = {
                    'data': [],
                    'ent_edge_index': [],
                    'raw_data': [],
                    'y': [],
                    'y_marker': [],
                }
                for modal_type in self.dataset['meta_data']['modal_types']:
                    self.dataset['data'][dataset_type][f'x_{modal_type}'] = []
                    self.dataset['data'][dataset_type][f'o11y_name_{modal_type}'] = []

                for i in range(len(self.ground_truth['time_interval'][dataset_type])):
                    for key in self.dataset['data'][dataset_type].keys():
                        if key != 'y_marker' and key != 'ent_edge_index' and 'x_' not in key:
                            self.dataset['data'][dataset_type][key].append([])
                    self.dataset['data'][dataset_type]['y_marker'].append(f'{self.ground_truth["entity_type"][dataset_type][i]}/{self.ground_truth["root_cause_type"][dataset_type][i]}')

                    date = self.ground_truth['time_interval'][dataset_type][i][0]
                    cloud_bed = self.ground_truth['time_interval'][dataset_type][i][1]
                    start_timestamp = self.ground_truth['time_interval'][dataset_type][i][2]
                    end_timestamp = self.ground_truth['time_interval'][dataset_type][i][3]

                    total_timestamps = self.raw_data['data'][dataset_type][f'{date}/{cloud_bed}']['node']['node-1']['timestamp'].tolist()
                    start_index, end_index = total_timestamps.index(start_timestamp), total_timestamps.index(end_timestamp)

                    self.dataset['data'][dataset_type]['ent_edge_index'].append(self.dataset['meta_data']['ent_edge_index'][dataset_type][f'{date}/{cloud_bed}'])

                    o11y_dict = {key: [] for key in self.dataset['meta_data']['modal_types']}
                    # preprocessed data, raw data.
                    for ent_type in self.raw_data['meta_data']['ent_types']:
                        ent_type_index_pair = self.raw_data['meta_data']['ent_type_index'][ent_type]
                        for ent_name in self.raw_data['meta_data']['ent_names'][ent_type_index_pair[0]:ent_type_index_pair[1]]:
                            data_df = self.raw_data['data'][dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name]
                            o11y_data = data_df.iloc[start_index:end_index, data_df.columns != "timestamp"]
                            self.dataset['data'][dataset_type]['data'][-1].append(o11y_data)
                            raw_data_df = self.raw_data['raw_data'][dataset_type][f'{date}/{cloud_bed}'][ent_type][ent_name]
                            self.dataset['data'][dataset_type]['raw_data'][-1].append(raw_data_df.iloc[start_index:end_index, raw_data_df.columns != "timestamp"])
                            index = 0
                            for modal_type in self.dataset['meta_data']['modal_types']:
                                o11y_dict[modal_type].extend(o11y_data.values[:, index:index + self.dataset['meta_data']['ent_feature_num'][ent_type][modal_type]].transpose(1, 0))
                                index += self.dataset['meta_data']['ent_feature_num'][ent_type][modal_type]
                                self.dataset['data'][dataset_type][f'o11y_name_{modal_type}'][-1].extend(self.dataset['meta_data']['o11y_names'][ent_type][modal_type])
                    for modal_type in self.dataset['meta_data']['modal_types']:
                        self.dataset['data'][dataset_type][f'x_{modal_type}'].append(np.array(o11y_dict[modal_type]))
                self.dataset['data'][dataset_type]['y'] = self.ground_truth['y'][dataset_type]

        index_dict = {
            'train': [],
            'valid': [],
            'unlabeled': []
        }
        if 'few_shot' in self.data_config['sample_strategy']:
            y_marker_set = set(self.dataset['data']['train_valid']['y_marker'])
            for y_marker in y_marker_set:
                sample_list = [index for (index, value) in enumerate(self.dataset['data']['train_valid']['y_marker']) if value == y_marker]
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
                        fill_list = np.random.choice(sample_list, self.data_config['each_train_size'] - len(sample_list)).tolist()
                        sample_list.extend(fill_list)
                    chosen_list = np.random.choice(sample_list, self.data_config['each_train_size'], replace=False).tolist()
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
        index_dict['unlabeled'] = list(set(range(len(self.dataset['data']['train_valid']['y_marker']))) - set(index_dict['train']) - set(index_dict['valid']))

        self.dataset['data']['unlabeled']['y_mask'] = [np.zeros(y.shape[0]) for y in self.dataset['data']['unlabeled']['y']]
        self.dataset['data']['unlabeled']['y_from_llm'] = [np.zeros(y.shape[0]) for y in self.dataset['data']['unlabeled']['y']]
        pkl_save(f'{save_file_path}/{file_name}', self.dataset)

    def update_dataset(self, k):
        save_file_path, file_name = self.get_dataset_file_name(k)
        # if not os.path.exists(f'{save_file_path}/{file_name}'):
        pkl_save(f'{save_file_path}/{file_name}', self.dataset)

    def predict_probs(self, data_loader, dataset_type):
        self.model.eval()
        y_pred = dict()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader[dataset_type]):
                out = self.model(batch_data)
                for ent_type in self.dataset['meta_data']['ent_types']:
                    if ent_type not in y_pred.keys():
                        y_pred[ent_type] = []
                    y_pred[ent_type].extend(torch.sigmoid(out[ent_type]).cpu().detach().numpy())
        return y_pred

    def evaluate_las_fault_localization(self, data_loader, dataset_type, threshold=0.5):
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
        all_pred, all_true = [], []
        fault_levels = dict()
        for ent_type in self.dataset['meta_data']['ent_types']:
            y_pred[ent_type], y_true[ent_type] = np.array(y_pred[ent_type]), np.array(y_true[ent_type])
            ent_index_pair = self.dataset['meta_data']['ent_type_index'][ent_type]
            y_pred[ent_type] = np.max(y_pred[ent_type].reshape(-1, ent_index_pair[1] - ent_index_pair[0], y_pred[ent_type].shape[1]), axis=2)
            y_true[ent_type] = np.max(y_true[ent_type].reshape(-1, ent_index_pair[1] - ent_index_pair[0], y_true[ent_type].shape[1]), axis=2)
            fault_levels[ent_type] = np.sum(y_true[ent_type], axis=1) == 1
            all_pred.append(y_pred[ent_type])
            all_true.append(y_true[ent_type])
        pred = np.concatenate(all_pred, axis=1)
        true = np.concatenate(all_true, axis=1)
        sort_pred = np.argsort(pred)[:, ::-1]

        result_dict = dict()
        k_list = [1, 3, 5]
        for ent_type in self.dataset['meta_data']['ent_types']:
            result_dict[ent_type] = {f'AC@{k}': 0 for k in k_list}
            sub_sort_pred, sub_true, sub_pred = sort_pred[fault_levels[ent_type]], true[fault_levels[ent_type]], pred[fault_levels[ent_type]]
            for k in k_list:
                for i in range(sub_sort_pred.shape[0]):
                    for j in range(k):
                        if sub_true[i][sub_sort_pred[i][j]] == 1:  # and sub_pred[i][sub_sort_pred[i][j]] > 0.5:
                            result_dict[ent_type][f'AC@{k}'] += 1
                            break
                result_dict[ent_type][f'AC@{k}'] /= sub_pred.shape[0]
        self.fl_result = copy.deepcopy(result_dict)
        logger.info('----------')
        logger.info(f'fault localization evaluation dataset type: {dataset_type}')
        for ent_type in self.dataset['meta_data']['ent_types']:
            logger.info(f'{ent_type.ljust(17)} | AC@1:  {result_dict[ent_type]["AC@1"]:.4f}; AC@3:  {result_dict[ent_type]["AC@3"]:.4f}; AC@5:  {result_dict[ent_type]["AC@5"]:.4f}')
        logger.info('----------')

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
