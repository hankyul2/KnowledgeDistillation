import os
import argparse

from src.train import run

parser = argparse.ArgumentParser(description='Knowledge Disillation')
parser.add_argument('-g', '--gpu_id', type=str, default='', help='specify which gpu you want to use')

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('DEVICE: {}'.format('cpu' if args.gpu_id == '' else args.gpu_id))

    run(args)
