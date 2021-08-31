import torch
from easydict import EasyDict as edict
from torch import nn
from torch.optim import SGD
import torch.optim.lr_scheduler as LR

from src.ModelWrapper import BaseModelWrapper
from src.at import AT
from src.fsp import FSP
from src.st import ST
from src.dataset import get_dataset, convert_to_dataloader
from src.resnet_32 import get_model

from src.utils import AverageMeter
from src.log import get_log_name, Result


class ModelWrapper(BaseModelWrapper):
    def __init__(self, log_name, model, teacher_model, device, criterion, kd_criterion, optimizer):
        super().__init__(log_name)
        self.model = model
        self.device = device
        self.criterion = criterion
        self.kd_criterion = kd_criterion
        self.optimizer = optimizer
        self.teacher_model = teacher_model

    def forward(self, x, y, epoch=None):
        std_conv1_out, std_act, std_feat, std_y_hat = self.model(x)
        teat_conv1_out, teat_act, teat_feat, teat_y_hat = self.teacher_model(x)

        if epoch < 100:
            cls_loss = torch.tensor(0.0).to(self.device)
            fsp_loss = self.kd_criterion([std_conv1_out] + std_act[:-1], [teat_conv1_out] + teat_act[:-1], std_act,
                                         teat_act)
        else:
            cls_loss = self.criterion(std_y_hat, y)
            fsp_loss = torch.tensor(0.0).to(self.device)

        loss = cls_loss + fsp_loss

        self.cls_losses.update(cls_loss.item(), x.size(0))
        self.fsp_losses.update(fsp_loss.item(), x.size(0))

        return loss, std_y_hat

    def init_progress(self, dl, epoch=None, mode='train'):
        super().init_progress(dl, epoch, mode)
        if mode == 'train':
            self.cls_losses = AverageMeter('CLS Loss', ':7.4f')
            self.fsp_losses = AverageMeter('FSP Loss', ':7.4f')
            self.progress.meters = [self.batch_time, self.data_time, self.losses, self.cls_losses,
                                    self.fsp_losses, self.top1, self.top5]


class MyOpt:
    def __init__(self, model, nbatch, phase1_lr=0.001, phase2_lr=0.1,  weight_decay=1e-4, momentum=0.9):
        self.optimizer = SGD(model.parameters(), lr=phase1_lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = LR.MultiStepLR(self.optimizer, milestones=[50, 75, 150, 175], gamma=0.1)
        self.phase2_lr = phase2_lr
        self.nbatch = nbatch
        self.step_ = 0
        self.nepoch = 0

    def step(self):
        self.optimizer.step()
        self.step_ += 1
        if self.step_ % self.nbatch == 0:
            self.scheduler.step()
            self.step_ = 0
            self.nepoch += 1
            if self.nepoch == 100:
                self.start_phase2()

    def start_phase2(self):
        for param in self.optimizer.param_groups:
            param['lr'] = self.phase2_lr

    def zero_grad(self):
        self.optimizer.zero_grad()


def run(args):
    # step 1. load dataset
    train_ds, valid_ds, test_ds = get_dataset(args.dataset)
    train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size,
                                              num_workers=args.num_workers, train=False)

    # step 2. load model (student, teacher)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_model = get_model(args.model_name, nclass=len(train_ds.classes), zero_init_residual=False).to(device)
    teacher_model = get_model(args.teacher_model, nclass=len(train_ds.classes), zero_init_residual=False,
                              pretrained_dataset=args.dataset).to(device)

    # step 3. prepare training tool
    criterion = nn.CrossEntropyLoss()
    kd_criterion = FSP()
    optimizer = MyOpt(model=student_model, nbatch=len(train_dl), phase2_lr=args.lr)

    # step 4. train
    model = ModelWrapper(log_name=get_log_name(args), model=student_model, teacher_model=teacher_model,
                         device=device, optimizer=optimizer, criterion=criterion, kd_criterion=kd_criterion)
    model.fit(train_dl, valid_dl, test_dl=None, nepoch=args.nepoch)

    # (extra) step 5. save result
    result_saver = Result()
    result_saver.save_result(args, model)


if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id': '',
    })
    run(args)
