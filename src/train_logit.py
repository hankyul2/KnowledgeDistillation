import os
from pathlib import Path

import torch
from easydict import EasyDict as edict
from torch import nn
from torch.optim import SGD
import torch.optim.lr_scheduler as LR

from src.ModelWrapper import BaseModelWrapper
from src.dataset import get_dataset, convert_to_dataloader
from src.resnet_32 import get_model

import numpy as np

class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, model, device, criterion, optimizer):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer


class MyOpt:
    def __init__(self, model, nbatch, lr=0.1, weight_decay=1e-4, momentum=0.9):
        self.optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = LR.MultiStepLR(self.optimizer, milestones=[100, 150], gamma=0.1)
        self.nbatch = nbatch
        self.step_ = 0

    def step(self):
        self.optimizer.step()
        self.step_ += 1
        if self.step_ % self.nbatch == 0:
            self.scheduler.step()
            self.step_ = 0

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_teacher_dataset(teacher_model, dataset, kd_method):
    model_path = {
        'resnet_32_20_cifar10': 'log/best_weight/2021-08-23/12-23-49-resnet_32_20_cifar10',
        'resnet_32_20_cifar100': 'log/best_weight/2021-08-23/12-23-48-resnet_32_20_cifar100',
        'resnet_32_110_cifar10': 'log/best_weight/2021-08-23/12-23-49-resnet_32_110_cifar10',
        'resnet_32_110_cifar100': 'log/best_weight/2021-08-23/12-23-49-resnet_32_110_cifar100',
    }
    teacher_model_path = model_path[teacher_model+'_'+dataset]
    base_path = os.path.join('kd', '_'.join(teacher_model_path.split('/')[-2:]))
    kd_np_path = os.path.join(base_path, kd_method + '.npy')
    Path(base_path).mkdir(parents=True, exist_ok=True)
    if os.path.exists(kd_np_path):
        kd_np = np.load(kd_np_path)
    else:
        kd_np = make_kd_np(teacher_model_path, kd_np_path, kd_method)

    return MyDataset(kd_np)


def run(args):
    # step 0. prepare teacher model output
    teacher_ds = get_teacher_dataset(args.teacher_model, args.dataset, args.kd_method)

    # # step 1. load dataset
    # train_ds, valid_ds, test_ds = get_dataset(args.dataset)
    # train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=True)
    # valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=False)
    #
    # # step 2. load model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = get_model(args.model_name, nclass=len(train_ds.classes), zero_init_residual=False).to(device)
    #
    # # step 3. prepare training tool
    # criterion = nn.CrossEntropyLoss()
    # optimizer = MyOpt(model=model, nbatch=len(train_dl), lr=args.lr)
    #
    # # step 4. train
    # model = ModelWrapper(args.model_name + '_' + args.dataset, model=model, device=device, optimizer=optimizer, criterion=criterion)
    # model.fit(train_dl, valid_dl, test_dl=None, nepoch=args.nepoch)

if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id':'',
    })
    run(args)
