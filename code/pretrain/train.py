import sys
sys.path.append('/workspace/project/working/2024/LasRCA/code')

import argparse
import time
from ts2vec import TS2Vec
from pretrain.data_loader import load_train_data
from shared_util.seed import *


def save_checkpoint_callback(
        save_every=1,
        unit='epoch'
):
    assert unit in ('epoch', 'iter')

    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')

    return callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='A_s/metric_trace_log')
    parser.add_argument('--dataset_path', default='/workspace/project/working/2024/LasRCA/temp_data/2022_CCF_AIOps_challenge/dataset/final/metric_trace_log.pkl')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=32, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=600, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=20, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, max_threads=args.max_threads)

    print('Loading data... ', end='')

    train_data = load_train_data(args.dataset, args.dataset_path)

    print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )

    if args.save_every is not None:
        unit = 'epoch'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = f'/workspace/project/working/2024/LasRCA/model/pretrain/{args.dataset}'
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')
    print("Finished.")
