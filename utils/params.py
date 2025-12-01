import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--dataset', type=str, default='data_2', help='dataset name',
                        choices=['data_1', 'data_2'])
    parser.add_argument('--epoch', default=800, type=int, help='number of epochs')
    parser.add_argument('--fold', default=5, type=int, help='number of fold')
    parser.add_argument('--query_num', type=int, help='Number of negative samples queried per round')
    parser.add_argument('--negative_sample_times', default=1, type=int, help='Ratio of negative to positive samples',
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument('--num_drug', type=float, help=' length')
    parser.add_argument('--num_protein', type=float, help=' length')
    parser.add_argument('--batch', default=10000, type=int, help='batch size')
    parser.add_argument('--tstbatch', default=10000, type=int, help='batch size')
    parser.add_argument('--latdim', default=128, type=int, help='embedding size',
                        choices=[32, 64, 128, 256])
    parser.add_argument('--seed', default=430, type=int, help='seed')
    parser.add_argument('--debug', default=True, type=bool, help='mode')

    return parser.parse_args()


args = ParseArgs()

params = {
    'data_1':
        {'k': 5,
         'lr': 5e-3,
         'weight_decay': 1e-3,
         'hlambda': 0.2,
         'reg': 0,
         'ssl_reg': 0.2,
         'latdim': 128,
         'gnn_layer': 3,
         'temp': 0.2,
         },

    'data_2':
        {'k': 5,
         'lr': 5e-3,
         'weight_decay': 1e-3,
         'hlambda': 0.2,
         'reg': 0,
         'ssl_reg': 0.2,
         'latdim': 128,
         'gnn_layer': 3,
         'temp': 0.2,
         },

}
