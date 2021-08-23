import time
import datetime
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils import AverageMeter, ProgressMeter, accuracy


class BaseModelWrapper:
    def __init__(self, log_name):
        self.best_acc = 0
        self.model = None
        self.device = None
        self.setup_directory()
        self.logger = logging.getLogger()
        self.log_name = '{}-{}'.format(datetime.datetime.now(), log_name)
        self.log_best_weight_path = 'log/best_weight/{}'.format(self.log_name)
        self.writer = SummaryWriter(log_dir='log/tensor_board/{}'.format(self.log_name), filename_suffix=log_name)
        self.setup_file_logger('log/text/{}-{}'.format(datetime.datetime.now(), log_name))


    def setup_directory(self):
        Path('log/text').mkdir(exist_ok=True, parents=True)
        Path('log/tensor_board').mkdir(exist_ok=True, parents=True)
        Path('log/best_weight').mkdir(exist_ok=True, parents=True)

    def setup_file_logger(self, log_file):
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log(self, message):
        print('{} {}'.format(datetime.datetime.now(), message))
        self.logger.info(message)

    def log_tensorboard(self, epoch, train_loss, train_acc, valid_loss, valid_acc):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/test', valid_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/test', valid_acc, epoch)

    def save_best_weight(self, model, top1_acc):
        if top1_acc > self.best_acc:
            self.best_acc = top1_acc
            self.log('Saving best model({:07.4f}%) weight to {}'.format(top1_acc, self.log_best_weight_path))
            torch.save({'weight':model.state_dict(), 'top1_acc':top1_acc}, self.log_best_weight_path)

    def load_best_weight(self, path=None):
        path = path if path else self.log_best_weight_path
        if self.model is not None and self.device is not None:
            self.model.load_state_dict(torch.load(f=path, map_location=self.device)["weight"])

    def train(self, train_dl, epoch):
        debug_step = len(train_dl)//10
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':7.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_dl),
            [batch_time, data_time, losses, top1, top5],
            prefix="TRAIN: [{}]".format(epoch))

        self.model.train()

        end = time.time()
        for step, (x, y) in enumerate(train_dl):
            data_time.update(time.time() - end)

            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)

            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
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

    @torch.no_grad()
    def valid(self, dl):
        debug_step = len(dl) // 10
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':7.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(dl),
            [batch_time, losses, top1, top5],
            prefix='VALID: ')

        self.model.eval()

        end = time.time()
        for step, (x, y) in enumerate(dl):
            x, y = x.float().to(self.device), y.long().to(self.device)
            y_hat = self.model.predict(x)
            loss = F.cross_entropy(y_hat, y)

            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if step % debug_step == 0:
                self.log(progress.display(step))

        return losses.avg, top1.avg

    def fit(self, train_dl, valid_dl, test_dl=None, nepoch=50):
        best_acc = 0
        best_acc_arg = 0

        for epoch in range(nepoch):
            train_loss, train_acc = self.train(train_dl, epoch)
            valid_loss, valid_acc = self.valid(valid_dl)

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_acc_arg = epoch + 1
                if test_dl:
                    self.evaluate(test_dl)
                self.save_best_weight(self.model, best_acc)

            print('=' * 150)
            self.log(
                '[EPOCH]: {:03d}/{:03d}    train loss {:07.4f} acc {:07.4f}%    valid loss {:07.4f} acc {:07.4f}% ('
                'best accuracy : {:07.4f} @ {:03d})'.format(
                    epoch + 1, nepoch, train_loss, train_acc, valid_loss, valid_acc, best_acc,
                    best_acc_arg
                ))
            print('=' * 150)
            self.log_tensorboard(epoch, train_loss, train_acc, valid_loss, valid_acc)

    @torch.no_grad()
    def evaluate(self, test_dl, ncrop=10):
        self.model.eval()
        total_loss = total_acc = total_y_hat = 0
        for iter in range(ncrop):
            y_hat = []
            total_y = []
            for x, y in test_dl:
                x, y = x.float().to(self.device), y.long().to(self.device)
                y_hat.append(self.model.predict(x))
                total_y.append(y)
            total_y_hat += torch.cat(y_hat, dim=0)
            total_y = torch.cat(total_y, dim=0)

        total_loss = F.cross_entropy(total_y_hat/ncrop, total_y).clone().detach().item()
        _, y_label = total_y_hat.max(dim=1)
        total_acc = (y_label == total_y).sum().item() / len(total_y)

        print('=' * 180)
        self.log('[EVALUATE] {}-crop loss {:07.4f}  |  {}-crop acc {:07.4f}%'.format(ncrop, total_loss, ncrop, total_acc * 100))
        print('=' * 180)