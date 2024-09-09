import sys
sys.path.append('/workspace/project/working/2024/LasRCA/code')

import argparse
import time
from ts2vec import TS2Vec
from shared_util.seed import *
from shared_util.common import *


def encode_dataset(dataset, dataset_path, model_config, model_device, epoch):
    result_dict = dict()

    model_base_path = '/workspace/project/working/2024/LasRCA/model/pretrain'
    repr_save_path = '/workspace/project/working/2024/LasRCA/temp_data/A/embeddings'
    raw_data = pkl_load(dataset_path)
    for ent_type in ['service', 'node', 'pod']:
        batch_process_dict = {
            'marker': [],
            'data': []
        }

        pretrained_weight_path = f'{model_base_path}/{dataset.replace("A", "A_" + ent_type[0])}/model{"" if epoch is None else f"_{epoch}"}.pkl'
        if not os.path.exists(pretrained_weight_path):
            continue

        input_dim = 0
        for dataset_type in ['normal', 'train_valid', 'test']:
            for date_cloud in raw_data['data'][dataset_type].keys():
                for ent in raw_data['data'][dataset_type][date_cloud][ent_type].keys():
                    df = raw_data['data'][dataset_type][date_cloud][ent_type][ent]
                    input_dim = np.array(df.iloc[:, df.columns != "timestamp"].values).shape[-1]

        pretrained_model = TS2Vec(
            input_dims=input_dim,
            device=model_device,
            **model_config
        )
        pretrained_model.load(pretrained_weight_path)
        for dataset_type in ['normal', 'train_valid', 'test']:
            if dataset_type not in result_dict.keys():
                result_dict[dataset_type] = dict()
            for date_cloud in raw_data['data'][dataset_type].keys():
                if date_cloud not in result_dict[dataset_type].keys():
                    result_dict[dataset_type][date_cloud] = dict()
                for ent in raw_data['data'][dataset_type][date_cloud][ent_type].keys():
                    if ent_type not in result_dict[dataset_type][date_cloud].keys():
                        result_dict[dataset_type][date_cloud][ent_type] = dict()
                    df = raw_data['data'][dataset_type][date_cloud][ent_type][ent]
                    batch_process_dict['marker'].append(f'{dataset_type}--{date_cloud}--{ent}')
                    batch_process_dict['data'].append(df.iloc[:, df.columns != "timestamp"].values)
        logger.info(f'{ent_type} start!')
        all_repr = pretrained_model.encode(np.array(batch_process_dict['data']), causal=True, sliding_length=1, sliding_padding=10)

        for i in range(len(batch_process_dict['marker'])):
            markers = batch_process_dict['marker'][i].split('--')
            result_dict[markers[0]][markers[1]][ent_type][markers[2]] = all_repr[i]
        logger.info(f'{ent_type} end!')
    os.makedirs(repr_save_path, exist_ok=True)
    pkl_save(f'{repr_save_path}/embedding{"" if epoch is None else f"_{epoch}"}.pkl', result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='A/metric_trace_log')
    parser.add_argument('--dataset_path', default='/workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_trace_log.pkl')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--batch-size', type=int, default=512)

    args = parser.parse_args()

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Arguments: {args}")

    config = dict(
        batch_size=args.batch_size,
        lr=0.001,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    device = init_dl_program(args.gpu, max_threads=args.max_threads)
    encode_dataset(args.dataset, args.dataset_path, config, device, args.epoch)

    print('done')
