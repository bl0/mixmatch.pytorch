import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


# noinspection PyAttributeOutsideInit
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CosineAnnealingLRWithRestart(CosineAnnealingLR):
    """Adjust learning rate"""

    def __init__(self, optimizer, eta_min=0, lr_t_0=10, lr_t_mul=2, last_epoch=-1):
        self.eta_min = eta_min
        self.lr_t_curr = lr_t_0
        self.lr_t_mul = lr_t_mul
        self.last_reset = 0
        super(CosineAnnealingLRWithRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_epoch = self.last_epoch - self.last_reset
        if curr_epoch >= self.lr_t_curr:
            self.lr_t_curr *= self.lr_t_mul
            self.last_reset = self.last_epoch
            rate = 0
        else:
            rate = curr_epoch * math.pi / self.lr_t_curr
        return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(rate))
                for base_lr in self.base_lrs]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         input: predictions for neural network
         target: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    losses = torch.sum(-target * nn.LogSoftmax(dim=1)(input), dim=1)
    if size_average:
        return torch.mean(losses)
    else:
        return torch.sum(losses)
