import sys

sys.path.append('/workspace/project/working/2024/LasRCA/code')

from tasks.base import *
import argparse
from shared_util.seed import *
from tasks.dataset import *
from tasks.util import *
from torch.utils.data import DataLoader
import copy
from collections import deque
from pprint import pprint


# cd /workspace/project/working/2024/LasRCA/code
# nohup python -u ./tasks/test/fault_type_classification_with_BCELogisticLoss.py > ./tasks/test/fault_type_classification_with_BCELogisticLoss_mix_up_5_5.log 2>&1 &


class BaseFaultTypeClassification(BaseClass):
    def __init__(self, data_config, model_config):
        super().__init__(data_config, model_config)

    def regular_load_data(self):
        data_loader = dict()
        for dataset_type in self.dataset['data'].keys():
            self.dataset['data'][dataset_type]['y'] = label_to_multi_class_format(self.dataset['data'][dataset_type]['y'], num_of_fault_types=len(self.dataset['meta_data']['fault_type_list']))
            if dataset_type == 'valid' or dataset_type == 'test':
                data_loader[dataset_type] = DataLoader(OriginalDataset(self.dataset['data'][dataset_type]), batch_size=self.model_config['batch_size'], shuffle=False)
            elif dataset_type == 'train':
                data_loader[dataset_type] = DataLoader(OriginalDataset(self.dataset['data'][dataset_type]), batch_size=self.model_config['batch_size'], shuffle=True)
            elif dataset_type == 'normal':
                data_loader[dataset_type] = DataLoader(OriginalDataset(self.dataset['data'][dataset_type]), batch_size=self.model_config['batch_size'], shuffle=True)
        return data_loader

    def regular_train(self, data_loader):
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
            train_loss /= len(data_loader['train'].dataset)
            logger.info(f'[{epoch}/{self.model_config["epochs"]}] | train_loss: {train_loss:.5f}')
        self.evaluate_las_fault_classification(data_loader, 'test')
        torch.save(self.model.state_dict(), f'{self.data_config["model_dir"]}/model_seed_{self.model_config["seed"]}.pkl')

    def fix_match_load_data(self):
        data_loader = dict()
        for dataset_type in self.dataset['data'].keys():
            self.dataset['data'][dataset_type]['y'] = label_to_multi_class_format(self.dataset['data'][dataset_type]['y'], num_of_fault_types=len(self.dataset['meta_data']['fault_type_list']))
            if dataset_type == 'valid' or dataset_type == 'test':
                data_loader[dataset_type] = DataLoader(OriginalDataset(self.dataset['data'][dataset_type]), batch_size=self.model_config['batch_size'], shuffle=False)
            data_loader['train'] = DataLoader(OriginalDataset(self.dataset['data']['train']), batch_size=self.model_config['batch_size'], shuffle=True)
            data_loader['unlabeled'] = DataLoader(UnlabeledAugmentedDataset(self.dataset['data']['unlabeled']), batch_size=self.model_config['batch_size'] * self.model_config['mu'], shuffle=True)
        return data_loader

    def mix_up_train(self, data_loader):
        labeled_train_iter = iter(data_loader['train'])
        for epoch in range(self.model_config['epochs']):
            self.model.train()
            train_loss, train_loss_x, train_loss_u = 0, 0, 0
            for batch_id in range(len(data_loader['train'])):
                self.optimizer.zero_grad()
                try:
                    labeled_batch_data = next(labeled_train_iter)
                except:
                    labeled_train_iter = iter(data_loader['train'])
                    labeled_batch_data = next(labeled_train_iter)
                batch_size = labeled_batch_data['y'].size(0)

                ent_index_dict = dict()
                for ent_type, index_pair in self.dataset['meta_data']['ent_type_index'].items():
                    ent_index_dict[ent_type] = index_pair[1] - index_pair[0]

                batch_data = {
                    key: torch.cat([labeled_batch_data[key]], dim=0) for key in [f'x_{modal_type}' for modal_type in self.dataset['meta_data']['modal_types']]
                }
                batch_data['ent_edge_index'] = torch.cat([labeled_batch_data['ent_edge_index']], dim=0)

                y_l_dict = rearrange_y(self.dataset['meta_data'], labeled_batch_data['y'], self.model_config['device'])
                y_dict = dict()
                for ent_type in self.dataset['meta_data']['ent_types']:
                    y_l_dict[ent_type] = y_l_dict[ent_type].view(batch_size, -1, y_l_dict[ent_type].shape[1])
                    y_dict[ent_type] = torch.cat([y_l_dict[ent_type]], dim=0)

                mix_x, mix_y = mix_up(labeled_batch_data, y_l_dict, alpha=self.model_config['alpha'], pseudo=False)

                final_mix_x = {
                    key: torch.cat([mix_x[key]], dim=0) for key in [f'x_{modal_type}' for modal_type in self.dataset['meta_data']['modal_types']]
                }
                final_mix_x['ent_edge_index'] = torch.cat([labeled_batch_data['ent_edge_index'], mix_x['ent_edge_index']], dim=0)
                final_mix_y = dict()
                for ent_type in self.dataset['meta_data']['ent_types']:
                    final_mix_y[ent_type] = torch.cat([y_dict[ent_type], y_l_dict[ent_type]], dim=0)

                mix_pred = self.model(mix_x)
                mix_pred = {ent_type: value.view(-1, ent_index_dict[ent_type], value.size(1)) for ent_type, value in mix_pred.items()}

                mix_pred_l, mix_pred_u = {key: value[:batch_size] for key, value in mix_pred.items()}, {key: value[2 * batch_size:] for key, value in mix_pred.items()}
                mix_y_l, mix_y_u = {key: value[:batch_size] for key, value in mix_y.items()}, {key: value[2 * batch_size:] for key, value in mix_y.items()}

                loss, loss_x, loss_u = 0, 0, 0
                # L_x
                for ent_type in self.dataset['meta_data']['ent_types']:
                    temp_loss = self.criterion[ent_type](mix_pred_l[ent_type].view(-1, mix_pred_l[ent_type].shape[2]), mix_y_l[ent_type].view(-1, mix_y_l[ent_type].shape[2]))
                    loss_x += temp_loss

                loss = loss_x
                train_loss += loss.item()
                train_loss_x += loss_x

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            logger.info(f'[{epoch}/{self.model_config["epochs"]}] | train_loss: {train_loss:.5f}, train_loss_x: {train_loss_x:.5f}')
        self.evaluate_las_fault_classification(data_loader, 'test')
        torch.save(self.model.state_dict(), f'{self.data_config["model_dir"]}/model_seed_{self.model_config["seed"]}.pkl')

    def fix_match_train(self, data_loader):
        labeled_train_iter, unlabeled_train_iter = iter(data_loader['train']), iter(data_loader['unlabeled'])

        p_deque = {ent_type: deque(maxlen=self.model_config['dbuf']) for ent_type in self.dataset['meta_data']['ent_types']}
        for epoch in range(self.model_config['epochs']):
            self.model.train()
            train_loss, train_loss_x, train_loss_u = 0, 0, 0
            for batch_id in range(len(data_loader['train'])):
                self.optimizer.zero_grad()
                try:
                    labeled_batch_data = next(labeled_train_iter)
                except:
                    labeled_train_iter = iter(data_loader['train'])
                    labeled_batch_data = next(labeled_train_iter)
                labeled_batch_size = labeled_batch_data['ent_edge_index'].size(0)
                try:
                    unlabeled_batch_data = unlabeled_train_iter.next()
                except:
                    unlabeled_train_iter = iter(data_loader['unlabeled'])
                    unlabeled_batch_data = next(unlabeled_train_iter)
                unlabeled_batch_size = unlabeled_batch_data['ent_edge_index'].size(0)
                unlabeled_batch_data_copy = copy.deepcopy(unlabeled_batch_data)

                ent_index_dict = dict()
                for ent_type, index_pair in self.dataset['meta_data']['ent_type_index'].items():
                    ent_index_dict[ent_type] = index_pair[1] - index_pair[0]

                mask = dict()
                with torch.no_grad():
                    pred_u = self.model(unlabeled_batch_data_copy)
                    y_u_dict = dict()
                    for ent_type in self.dataset['meta_data']['ent_types']:
                        p = torch.sigmoid(pred_u[ent_type]).detach()
                        if len(p_deque[ent_type]) == self.model_config['dbuf']:
                            moving_average_data = torch.stack(list(p_deque[ent_type]))
                            moving_average_data = moving_average_data.reshape(moving_average_data.shape[0] * moving_average_data.shape[1], moving_average_data.shape[2])
                            p_hat = torch.mean(moving_average_data, 0)

                            new_positive = p * (1 / p_hat)
                            new_negative = (1 - p) * (self.ent_fault_type_weight[ent_type] / (1 - p_hat))
                            final_p = new_positive / (new_positive + new_negative)
                        else:
                            final_p = p
                        mask[ent_type] = torch.logical_or(torch.all(p < self.model_config['tau_boundary'], dim=1), torch.any(p > (1 - self.model_config['tau_boundary']), dim=1))
                        # mask[ent_type] = torch.any(p > self.model_config['tau_boundary'], dim=1)
                        y_u_dict[ent_type] = (final_p > (1 - self.model_config['tau_boundary'])).to(float)
                        p_deque[ent_type].append(p)

                y_l_dict = rearrange_y(self.dataset['meta_data'], labeled_batch_data['y'], self.model_config['device'])
                y_dict = dict()
                for ent_type in self.dataset['meta_data']['ent_types']:
                    y_u_dict[ent_type] = y_u_dict[ent_type].view(unlabeled_batch_size, -1, y_u_dict[ent_type].shape[1])
                    y_l_dict[ent_type] = y_l_dict[ent_type].view(labeled_batch_size, -1, y_l_dict[ent_type].shape[1])
                    y_dict[ent_type] = torch.cat([y_l_dict[ent_type], y_u_dict[ent_type], y_u_dict[ent_type]], dim=0)

                mix_x_l, mix_y_l = mix_up(labeled_batch_data, y_l_dict, alpha=self.model_config['alpha'], pseudo=False)
                mix_x_u, mix_y_u = mix_up(unlabeled_batch_data, y_u_dict, alpha=self.model_config['alpha'], pseudo=True)

                mix_pred_l = self.model(mix_x_l)
                mix_pred_l = {ent_type: value.view(-1, ent_index_dict[ent_type], value.size(1)) for ent_type, value in mix_pred_l.items()}

                mix_pred_u = self.model(mix_x_u)
                mix_pred_u = {ent_type: value.view(-1, ent_index_dict[ent_type], value.size(1)) for ent_type, value in mix_pred_u.items()}

                loss, loss_x, loss_u = 0, 0, 0
                for ent_type in self.dataset['meta_data']['ent_types']:
                    # L_x
                    temp_loss = self.criterion[ent_type](mix_pred_l[ent_type].view(-1, mix_pred_l[ent_type].shape[2]), mix_y_l[ent_type].view(-1, mix_y_l[ent_type].shape[2]))
                    loss_x += temp_loss

                    # L_u
                    if len(p_deque[ent_type]) == self.model_config['dbuf']:
                        temp_loss = (self.u_criterion[ent_type](mix_pred_u[ent_type].view(-1, mix_pred_u[ent_type].shape[2]), mix_y_u[ent_type].view(-1, mix_y_u[ent_type].shape[2])) * mask[ent_type].reshape(mask[ent_type].size(0), 1)).mean()
                        loss_u += temp_loss

                loss = loss_x + self.model_config['lambda_u'] * linear_ramp_up(epoch, self.model_config['epochs']) * loss_u
                train_loss += loss.item()
                train_loss_x += loss_x
                train_loss_u += loss_u

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            logger.info(f'[{epoch}/{self.model_config["epochs"]}] | train_loss: {train_loss:.5f}, train_loss_x: {train_loss_x:.5f}, train_loss_u: {train_loss_u:.5f}')
        self.evaluate_las_fault_localization(data_loader, 'test')
        self.evaluate_las_fault_classification(data_loader, 'test')
        torch.save(self.model.state_dict(), f'{self.data_config["model_dir"]}/model_seed_{self.model_config["seed"]}.pkl')

    def train(self):
        if self.model_config['strategy'] == 'regular':
            data_loader = self.regular_load_data()
            self.regular_train(data_loader)
        elif self.model_config['strategy'] == 'mix_up':
            data_loader = self.regular_load_data()
            self.mix_up_train(data_loader)
        elif self.model_config['strategy'] == 'fix_match':
            data_loader = self.fix_match_load_data()
            self.fix_match_train(data_loader)

    def final_test(self):
        if self.model_config['strategy'] == 'regular' or self.model_config['strategy'] == 'mix_up':
            data_loader = self.regular_load_data()
        else:
            data_loader = self.fix_match_load_data()
        self.model.load_state_dict(torch.load(f'{self.data_config["model_dir"]}/model_seed_{self.model_config["seed"]}.pkl'))
        self.evaluate_las_fault_localization(data_loader, 'test')
        self.evaluate_las_fault_classification(data_loader, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    window_size = 9
    parser.add_argument('--dataset', default='A/metric_trace_log')
    parser.add_argument('--dataset_path', default='2022_CCF_AIOps_challenge/dataset/final/metric_trace_log.pkl')
    parser.add_argument('--ground_truth_path', default=f'2022_CCF_AIOps_challenge/dataset/time_interval_and_label/time_interval_window_size_{window_size}.pkl')

    parser.add_argument('--sample_strategy', default='narrow_few_shot', help='random or narrow_few_shot or broad_few_shot')
    parser.add_argument('--train_size', type=int, default=200)
    parser.add_argument('--valid_size', type=int, default=60)
    parser.add_argument('--each_train_size', type=int, default=1)
    parser.add_argument('--each_valid_size', type=int, default=1)
    parser.add_argument('--window_size', default=window_size)

    parser.add_argument('--strategy', type=str, default='fix_match')

    parser.add_argument('--seed', type=int, default=405, help='Random seed for deterministic')
    parser.add_argument('--k', type=int, default=0, help='The k-th retraining.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--max_threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size (defaults to 8)')

    parser.add_argument('--epochs', type=int, default=500, help='The number of epochs')
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)

    # MixMatch params
    parser.add_argument('--optimizer', default='Adam', help='Adam or SGD')
    parser.add_argument('--T', default=0.5, help='Sharpening Temperature')
    parser.add_argument('--alpha', type=float, default=0.2, help='MixUp beta hyperparameter')
    parser.add_argument('--mu', type=float, default=1, help='Unlabeled batch size ratio, 1 for MixMatch and 7 for FixMatch')
    parser.add_argument('--lambda_u', type=float, default=0.1, help='Unlabeled loss weight')
    parser.add_argument('--tau_boundary', type=float, default=0.05, help='Pseudo label threshold boundary')
    parser.add_argument('--dbuf', type=int, default=4, help='Num of batches for distribution alignment.')

    # O11y representation learning params
    parser.add_argument("--orl_te_heads", default=2, type=int)
    parser.add_argument("--orl_te_layers", default=2, type=int)
    parser.add_argument("--orl_te_in_channels", default=256, type=int)

    # RE feature integration params
    parser.add_argument("--efi_in_dim", default=256, type=int)
    parser.add_argument("--efi_te_heads", default=4, type=int)
    parser.add_argument("--efi_te_layers", default=2, type=int)
    parser.add_argument("--efi_out_dim", default=64 * 4, type=int)

    # RE feature fusion params
    parser.add_argument("--eff_in_dim", default=64 * 4, type=int)
    parser.add_argument("--eff_GAT_out_channels", default=128, type=int)
    parser.add_argument("--eff_GAT_heads", default=2, type=int)
    parser.add_argument("--eff_GAT_dropout", default=0.1, type=float)

    args = parser.parse_args()

    device = init_dl_program(args.gpu, max_threads=args.max_threads)

    args = vars(args)

    seed_everything(seed=args["seed"])
    args['device'] = device
    model_dir = f'/workspace/project/working/2024/LasRCA/model/{args["strategy"]}/{args["dataset"]}'
    os.makedirs(model_dir, exist_ok=True)
    test_data_config = {
        'dataset': f'{args["dataset"]}',
        'dataset_path': f'{args["dataset_path"]}',
        'ground_truth_path': f'{args["ground_truth_path"]}',
        'model_dir': f'{model_dir}',
        'sample_strategy': args['sample_strategy'],
        'train_size': args['train_size'],
        'valid_size': args['valid_size'],
        'each_train_size': args['each_train_size'],
        'each_valid_size': args['each_valid_size'],
        'window_size': args['window_size']
    }
    # fault_type_classification = BaseFaultTypeClassification(test_data_config, args)
    # fault_type_classification.train()
    # fault_type_classification.prepare_dataset()
    # fault_type_classification.final_test()

    final_dict = dict()
    for seed in [405, 406, 407, 408, 409]:
        args['seed'] = seed
        fault_type_classification = BaseFaultTypeClassification(test_data_config, args)
        fault_type_classification.final_test()
        for result in [fault_type_classification.fl_result, fault_type_classification.fc_result]:
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
    pprint(final_dict)
    for ent_type in final_dict.keys():
        for em in final_dict[ent_type].keys():
            print(f'{ent_type} | {em} | mean: {np.mean(final_dict[ent_type][em])}; std: {np.std(final_dict[ent_type][em])}')

    ...
