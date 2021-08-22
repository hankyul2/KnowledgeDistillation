import torch
from easydict import EasyDict as edict
from torch import nn
from torch.optim import SGD
import torch.optim.lr_scheduler as LR

from src.ModelWrapper import BaseModelWrapper
from src.dataset import get_dataset, convert_to_dataloader
from src.resnet_32 import get_model


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, model, device, criterion, optimizer):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer


class MyOpt:
    def __init__(self, model, nbatch, lr=0.1, nepoch=200, weight_decay=1e-4, momentum=0.9):
        self.optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = LR.StepLR(self.optimizer, nbatch * (nepoch / 3), 0.1)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def run(args):
    # step 1. load dataset
    train_ds, valid_ds, test_ds = get_dataset(args.dataset)
    train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=False)

    # step 2. load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_name, nclass=len(train_ds.classes), zero_init_residual=True).to(device)

    # step 3. prepare training tool
    criterion = nn.CrossEntropyLoss()
    optimizer = MyOpt(model=model, nbatch=len(train_dl), lr=args.lr)

    # step 4. train
    model = ModelWrapper(args.model_name + '_' + args.dataset, model=model, device=device, optimizer=optimizer, criterion=criterion)
    model.fit(train_dl, valid_dl, test_dl=None, nepoch=args.nepoch)

if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id':'',
    })
    run(args)
