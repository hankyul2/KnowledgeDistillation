import time

import torch
from easydict import EasyDict as edict
from torch import nn
from torch.optim import SGD
import torch.optim.lr_scheduler as LR

from src.ModelWrapper import BaseModelWrapper
from src.dataset import get_dataset, convert_to_dataloader
from src.logits import Logits
from src.resnet_32 import get_model

from src.utils import AverageMeter, ProgressMeter, accuracy


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, model, teacher_model, device, criterion, kd_criterion, optimizer):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.kd_criterion = kd_criterion
        self.optimizer = optimizer
        self.teacher_model = teacher_model

    def train(self, train_dl, epoch):
        debug_step = len(train_dl)//10
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Total Loss', ':7.4f')
        cls_losses = AverageMeter('CLS Loss', ':7.4f')
        kd_losses = AverageMeter('KD Loss', ':7.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_dl),
            [batch_time, data_time, losses, cls_losses, kd_losses, top1, top5],
            prefix="TRAIN: [{}]".format(epoch))

        self.model.train()

        end = time.time()
        for step, (x, y) in enumerate(train_dl):
            data_time.update(time.time() - end)

            x, y = x.to(self.device), y.to(self.device)
            _, std_feat, std_y_hat = self.model(x)
            _, teat_feat, _ = self.teacher_model(x)
            cls_loss = self.criterion(std_y_hat, y)
            kd_loss = self.kd_criterion(std_feat, teat_feat.detach())
            loss = cls_loss + kd_loss

            acc1, acc5 = accuracy(std_y_hat, y, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            cls_losses.update(cls_loss.item(), x.size(0))
            kd_losses.update(kd_loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if step != 0 and step % debug_step == 0:
                self.log(progress.display(step))

        return losses.avg, top1.avg


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


def run(args):
    # step 1. load dataset
    train_ds, valid_ds, test_ds = get_dataset(args.dataset)
    train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=False)

    # step 2. load model (student, teacher)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_model = get_model(args.model_name, nclass=len(train_ds.classes), zero_init_residual=False).to(device)
    teacher_model = get_model(args.teacher_model, nclass=len(train_ds.classes), zero_init_residual=False,
                              pretrained=args.teacher_model+'_'+args.dataset).to(device)

    # step 3. prepare training tool
    criterion = nn.CrossEntropyLoss()
    kd_criterion = Logits()
    optimizer = MyOpt(model=student_model, nbatch=len(train_dl), lr=args.lr)

    # step 4. train
    model = ModelWrapper(args.model_name + '_' + args.dataset + '_' + args.teacher_model + '_' + args.kd_method, model=student_model,
                         teacher_model=teacher_model, device=device, optimizer=optimizer, criterion=criterion, kd_criterion=kd_criterion)
    model.fit(train_dl, valid_dl, test_dl=None, nepoch=args.nepoch)

if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id':'',
    })
    run(args)
