import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='./semi-data3',
                        help='the path of training data')

    parser.add_argument('--val_path',
                        type=str,
                        default='./val',
                        help='the path of val data')
    
    parser.add_argument('--test_path',
                        type=str,
                        default='./test',
                        help='the path of test data')
    parser.add_argument('--save_path',
                        type=str,
                        default='./checkpoints',
                        help='the path of save_model')
    parser.add_argument('--deterministic',
                        type=int,
                        default=0,
                        help='whether use deterministic training')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='maximum iterations number to train')

    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-5,
                        help='maximum epoch number to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='the batch_size of training size')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.99,
                        help='ema_decay')
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='alpha of supervised loss para')
    parser.add_argument('--beta',
                        type=float,
                        default=1.0,
                        help='beta of weak supervised loss para')

    parser.add_argument('--consistency',
                        type=float,
                        default=0.1,
                        help='consistency')
    parser.add_argument('--consistency_rampup',
                        type=float,
                        default=40.0,
                        help='consistency_rampup')
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help='Use mixed precision')

    args = parser.parse_args()
    return args
