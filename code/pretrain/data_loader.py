import numpy as np
from shared_util.common import *


def load_train_data(dataset, dataset_path):
    raw_data = pkl_load(dataset_path)
    train_data = []
    if 'A' in dataset:
        ent_type = ''
        if 'A_s' in dataset:
            ent_type = 'service'
        elif 'A_n' in dataset:
            ent_type = 'node'
        elif 'A_p' in dataset:
            ent_type = 'pod'
        for dataset_type in ['normal', 'train_valid']:
            for date_cloud in raw_data['data'][dataset_type].keys():
                for ent in raw_data['data'][dataset_type][date_cloud][ent_type].keys():
                    df = raw_data['data'][dataset_type][date_cloud][ent_type][ent]
                    train_data.append(np.array(df.iloc[:, df.columns != "timestamp"].values))
        train_data = np.array(train_data)
        return train_data
