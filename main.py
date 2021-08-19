import os
import argparse
import numpy as np
import random

import torch

from src.train import run

parser = argparse.ArgumentParser(description='Knowledge Disillation')
parser.add_argument('-g', '--gpu_id', type=str, default='', help='specify which gpu you want to use')
parser.add_argument('-s', '--seed', type=int, default=3, help='fix random seed')


def init(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('DEVICE: {}'.format('cpu' if args.gpu_id == '' else args.gpu_id))


if __name__ == '__main__':
    args = parser.parse_args()
    init(args)

    run(args)
