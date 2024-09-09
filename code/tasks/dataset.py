from torch.utils.data import Dataset
import torch
import numpy as np
from tsaug import TimeWarp, AddNoise


class OriginalDataset(Dataset):
    """
    加载原始数据的Dataset, 包括原始数据和标签.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        item = dict()
        for key in self.data.keys():
            if 'x_' in key:
                item[key] = torch.FloatTensor(np.array(self.data[key][idx]).astype(np.float64))
            if key == 'ent_edge_index':
                item[key] = torch.LongTensor(self.data[key][idx])
            if key == 'y':
                item[key] = torch.FloatTensor(np.array(self.data[key][idx]).astype(np.float64))
        return item


class UnlabeledDataset(Dataset):
    """
    生成PyTorch加载数据需要的Dataset, 不进行任何变换.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        item = dict()
        for key in self.data.keys():
            if 'x_' in key:
                item[key] = torch.FloatTensor(np.array(self.data[key][idx]).astype(np.float64))
            if key == 'ent_edge_index':
                item[key] = torch.LongTensor(self.data[key][idx])
            item['y'] = torch.FloatTensor(np.array(self.data['y'][idx]).astype(np.float64))
            item['y_mask'] = torch.FloatTensor(np.array(self.data['y_mask'][idx]).astype(np.float64))
            item['y_true'] = torch.FloatTensor(np.array(self.data['y'][idx]).astype(np.float64))
        return item


class LabeledAugmentedDataset(Dataset):
    """
    通过时间序列扭曲和变换生成增强的有标签Dataset.
    """
    def __init__(self, data):
        self.data = data
        self.augmenter = AddNoise(scale=0.01) @ 0.8  # (TimeWarp() * 1 + AddNoise() @ 0.8)

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        item = dict()
        for key in self.data.keys():
            if 'x_' in key:
                augmented_data = self.augmenter.augment(np.expand_dims(np.array(self.data[key][idx]).transpose(1, 0), axis=0)).transpose(0, 2, 1).squeeze(axis=0)
                item[key] = torch.FloatTensor(augmented_data.astype(np.float64))
            if key == 'ent_edge_index':
                item[key] = torch.LongTensor(self.data[key][idx])
            if key == 'y':
                item[key] = torch.FloatTensor(np.array(self.data[key][idx]).astype(np.float64))
        return item


class UnlabeledAugmentedDataset(Dataset):
    """
    通过时间序列扭曲和变换生成增强的无标签Dataset.
    """
    def __init__(self, data):
        self.data = data
        self.augmenter = (AddNoise(scale=0.01) @ 0.8)

    def __len__(self):
        return len(self.data['y'])

    # MixMatch version
    # def __getitem__(self, idx):
    #     item = dict()
    #     for key in self.data.keys():
    #         if 'x_' in key:
    #             augmented_data_1 = self.augmenter.augment(np.expand_dims(np.array(self.data[key][idx]).transpose(1, 0), axis=0)).transpose(0, 2, 1).squeeze(axis=0)
    #             augmented_data_2 = self.augmenter.augment(np.expand_dims(np.array(self.data[key][idx]).transpose(1, 0), axis=0)).transpose(0, 2, 1).squeeze(axis=0)
    #             item[f'{key}_1'] = torch.FloatTensor(augmented_data_1.astype(np.float64))
    #             item[f'{key}_2'] = torch.FloatTensor(augmented_data_2.astype(np.float64))
    #         if key == 'ent_edge_index':
    #             item[key] = torch.LongTensor(self.data[key][idx])
    #     return item

    # FixMatch version
    def __getitem__(self, idx):
        item = dict()
        for key in self.data.keys():
            if 'x_' in key:
                augmented_data = self.augmenter.augment(np.expand_dims(np.array(self.data[key][idx]).transpose(1, 0), axis=0)).transpose(0, 2, 1).squeeze(axis=0)
                item[f'{key}'] = torch.FloatTensor(augmented_data.astype(np.float64))
            if key == 'ent_edge_index':
                item[key] = torch.LongTensor(self.data[key][idx])

            # only for testing
            if key == 'y':
                item[key] = torch.FloatTensor(np.array(self.data[key][idx]).astype(np.float64))
        return item
