import os
import argparse
import numpy as np
import random

import torch

parser = argparse.ArgumentParser(description='Knowledge Disillation')
parser.add_argument('-g', '--gpu_id', type=str, default='', help='Enter which gpu you want to use')
parser.add_argument('-s', '--seed', type=int, default=None, help='Enter random seed')
parser.add_argument('-m', '--model_name', type=str, default='resnet_32_20', help='Enter model name')
parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Enter dataset')
parser.add_argument('-b', '--batch_size', type=int, default=128, help='Enter batch size for train step')
parser.add_argument('-w', '--num_workers', type=int, default=4, help='Enter the number of workers per dataloader')
parser.add_argument('-l', '--lr', type=float, default=0.1, help='Enter learning rate')
parser.add_argument('-e', '--nepoch', type=int, default=200, help='Enter the number of epoch')
parser.add_argument('-t', '--teacher_model', type=str, default='resnet_32_20', help='Enter teacher model name')
parser.add_argument('-k', '--kd_method', type=str, default='base', choices=[
    'base', 'logit', 'st'
], help='Enter knowledge Distillation Method')


def init(args):
    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('DEVICE: {}'.format('cpu' if args.gpu_id == '' else args.gpu_id))


if __name__ == '__main__':
    args = parser.parse_args()
    init(args)

    if args.kd_method == 'base':
        from src.train import run
    elif args.kd_method == 'logit':
        from src.train_logit import run
    elif args.kd_method == 'st':
        from src.train_st import run

    run(args)
