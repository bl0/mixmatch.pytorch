"""Train CIFAR10 with PyTorch."""
import argparse
import os
import random
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from lib.datasets.cifar import UnlabeledCIFAR10, PseudoCIFAR10
from lib.models import WideResNet, resnet18_cifar
from lib.utils import AverageMeter, accuracy, CosineAnnealingLRWithRestart,  cross_entropy
from test import validate


def get_dataloader(args):
    normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616))
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    args.ndata = 50000
    num_labeled_data = args.num_labeled
    torch.manual_seed(args.rng_seed)
    perm = torch.randperm(args.ndata)
    labeled_indexes = perm[:num_labeled_data]
    unlabeled_indexes = perm[num_labeled_data:]

    trainset = PseudoCIFAR10(
            labeled_indexes=labeled_indexes, root=args.data_dir,
            train=True, transform=transform_train)

    trainloder = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, drop_last=True,
            shuffle=True, num_workers=args.num_workers)

    unlabeled_trainset = UnlabeledCIFAR10(
            indexes=unlabeled_indexes, nu=2,
            root='./data', transform=transform_train)

    unlabeled_trainloder = torch.utils.data.DataLoader(
        unlabeled_trainset, batch_size=args.batch_size, drop_last=True,
        shuffle=True, num_workers=args.num_workers)

    testset = CIFAR10(root=args.data_dir, train=False,
                      download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    print('-'*80)
    print('selected labeled indexes: ', labeled_indexes)

    return testloader, trainloder, unlabeled_trainloder


def build_model(args):
    if args.architecture == 'resnet18':
        net = resnet18_cifar(low_dim=args.num_class)
    elif args.architecture.startswith('wrn'):
        split = args.architecture.split('-')
        net = WideResNet(depth=int(split[1]), widen_factor=int(split[2]),
                         num_classes=args.num_class)
    else:
        raise ValueError('architecture should be resnet18 or wrn')
    net = net.to(args.device)

    print('#param: {}'.format(sum([p.nelement() for p in net.parameters()])))

    if args.device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # resume from unsupervised pretrain
    if len(args.resume) > 0:
        # Load checkpoint.
        print('==> Resuming from unsupervised pretrained checkpoint..')
        checkpoint = torch.load(args.resume)
        # only load shared conv layers, don't load fc
        model_dict = net.state_dict()
        pretrained_dict = checkpoint['net']
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                if k in model_dict
                                and v.size() == model_dict[k].size()}
        assert len(pretrained_dict) > 0
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    return net


def get_lr_scheduler(optimizer, lr_scheduler, max_iters):
    if lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, max_iters, eta_min=0.00001)
    elif lr_scheduler == 'cosine-with-restart':
        scheduler = CosineAnnealingLRWithRestart(optimizer, eta_min=0.00001)
    else:
        raise ValueError("not supported")

    return scheduler


def guess_label(ys, net, T=0.5):
    net = net.eval()
    logits = net(torch.cat(ys, dim=0))

    p = torch.nn.functional.softmax(logits, dim=-1).view(len(ys), -1, logits.shape[1])
    p_target = p.mean(dim=0).pow(1. / T)
    p_target /= p_target.sum(dim=1, keepdim=True)

    return p_target

def mixup(x, l, beta=0.5):
    assert x.shape[0] == l.shape[0]
    mix = torch.distributions.Beta(beta, beta).sample((x.shape[0], )).to(x.device).view(-1, 1, 1, 1)

    mix = torch.max(mix, 1 - mix)
    perm = torch.randperm(x.shape[0])

    xmix = x * mix + x[perm] * (1 - mix)
    lmix = l * mix[..., 0, 0] + l[perm] * (1 - mix[..., 0, 0])

    return xmix, lmix

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

# Training
def train(net, optimizer, scheduler, trainloader, unlabeled_trainloder, testloader, summary_writer, args):
    train_loss = AverageMeter()
    train_xe_loss = AverageMeter()
    train_l2u_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    best_acc = 0
    end = time.time()
    def inf_generator(trainloader, unlabeled_trainloder):
        while True:
            for data in zip(trainloader, unlabeled_trainloder):
                yield data

    for step, ((x_in, l_in), y_in) in enumerate(inf_generator(trainloader, unlabeled_trainloder)):
        if step >= args.max_iters:
            break
        data_time.update(time.time() - end)

        with torch.no_grad():
            x_in = x_in.to(args.device)
            l_in = l_in.to(args.device)
            y_in = [yi.to(args.device) for yi in y_in]
            guess = guess_label(y_in, net)

            nu = len(y_in)
            bs = x_in.shape[0]
            assert bs == y_in[0].shape[0]

            # mixup
            l_in_onehot = torch.zeros(bs, args.num_class).float().to(args.device)
            l_in_onehot[np.arange(bs), l_in] = 1
            xy, l_xy = mixup(torch.cat([x_in] + y_in, dim=0),
                             torch.cat([l_in_onehot] + [guess] * nu, dim=0))
            xy = xy.reshape([nu+1] + list(x_in.shape))
            l_xy = l_xy.reshape([nu+1] + list(l_in_onehot.shape))
            x, y = xy[0], xy[1:]
            l_x, l_y = l_xy[0], l_xy[1:]

        # switch to train mode
        net.train()
        # scheduler.step()
        optimizer.zero_grad()

        # forward
        batches = interleave([x, y[0], y[1]], bs)
        logits = [net(batches[0])]

        for batchi in batches[1:]:
            logits.append(net(batchi))
        logits = interleave(logits, bs)
        logits_x = logits[0]
        logits_y = torch.cat(logits[1:], 0)

        # loss
        loss_xe = cross_entropy(input=logits_x, target=l_x)
        loss_l2u = nn.functional.mse_loss(nn.functional.softmax(logits_y, dim=-1), l_y.reshape(nu * bs, args.num_class))
        w_match = 100 * min(1, step / 8192.0)
        loss = loss_xe + w_match * loss_l2u

        prec1, = accuracy(logits_x, l_in)
        top1.update(prec1[0], x_in.size(0))

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), x_in.size(0))
        train_xe_loss.update(loss_xe.item(), x_in.size(0))
        train_l2u_loss.update(loss_l2u.item(), y_in[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        summary_writer.add_scalar('w_match', w_match, step)
        summary_writer.add_scalar('top1', top1.val, step)
        summary_writer.add_scalar('loss/all', train_loss.val, step)
        summary_writer.add_scalar('loss/xe', train_xe_loss.val, step)
        summary_writer.add_scalar('loss/l2u', train_l2u_loss.val, step)

        if step % args.print_freq == 0:
            print(f'Train: [{step}/{args.max_iters}] '
                  f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'prec1: {top1.val:.3f} ({top1.avg:.3f}) '
                  f'Loss_xe: {train_xe_loss.val:.3f} ({train_xe_loss.avg:.3f}) '
                  f'Loss_l2u: {train_l2u_loss.val:.4f} ({train_l2u_loss.avg:.4f})')

        if (step + 1) % args.eval_freq == 0 or step == args.max_iters - 1:
            acc = validate(testloader, net,
                           device=args.device, print_freq=args.print_freq)

            summary_writer.add_scalar('val_top1', acc, step)

            if acc > best_acc:
                best_acc = acc
                state = {
                    'step': step,
                    'best_acc': best_acc,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                os.makedirs(args.model_dir, exist_ok=True)
                torch.save(state, os.path.join(args.model_dir, 'ckpt.pth.tar'))

            print('best accuracy: {:.2f}\n'.format(best_acc))

def main(args):
    # Data
    print('==> Preparing data..')
    testloader, trainloder, unlabeled_trainset = get_dataloader(args)

    print('==> Building model..')
    net = build_model(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.02)

    scheduler = get_lr_scheduler(optimizer, args.lr_scheduler, args.max_iters)

    if args.eval:
        return validate(testloader, net,
                        device=args.device, print_freq=args.print_freq)
    # summary writer
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(args.log_dir)

    train(net, optimizer, scheduler,
          trainloder, unlabeled_trainset, testloader, summary_writer, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', '--dataDir', default='./data',
                        type=str, metavar='DIR')
    parser.add_argument('--model-root', default='./checkpoint/cifar10-semi',
                        type=str, metavar='DIR',
                        help='root directory to save checkpoint')
    parser.add_argument('--log-root', default='./tensorboard/cifar10-semi',
                        type=str, metavar='DIR',
                        help='root directory to save tensorboard logs')
    parser.add_argument('--exp-name', default='exp', type=str,
                        help='experiment name, used to determine log_dir and model_dir')
    parser.add_argument('--lr', default=0.0001, type=float,
                        metavar='LR', help='learning rate')
    parser.add_argument('--lr-scheduler', default='cosine', type=str,
                        choices=['multi-step', 'cosine', 'cosine-with-restart'],
                        help='which lr scheduler to use')
    parser.add_argument('--resume', '-r', default='', type=str,
                        metavar='FILE', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='test only')
    parser.add_argument('--finetune', action='store_true',
                        help='only training last fc layer')
    parser.add_argument('-j', '--num-workers', default=2, type=int,
                        metavar='N', help='number of workers to load data')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='batch size')
    parser.add_argument('--max-iters', default=500000, type=int,
                        metavar='N', help='number of iterations')
    parser.add_argument('--num-labeled', default=250, type=int,
                        metavar='N', help='number of labeled data')
    parser.add_argument('--rng-seed', default=0, type=int,
                        metavar='N', help='random number generator seed')
    parser.add_argument('--gpus', default='0', type=str, metavar='GPUS')
    parser.add_argument('--eval-freq', default=500, type=int,
                        metavar='N', help='eval frequence')
    parser.add_argument('--print-freq', default=100, type=int,
                        metavar='N', help='print frequence')
    parser.add_argument('--architecture', '--arch', default='wrn-28-2', type=str,
                        help='which backbone to use')
    opt, rest = parser.parse_known_args()
    print(rest)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.num_class = 10
    opt.log_dir = os.path.join(opt.log_root, opt.exp_name)
    opt.model_dir = os.path.join(opt.model_root, opt.exp_name)

    torch.manual_seed(opt.rng_seed)
    torch.cuda.manual_seed(opt.rng_seed)
    random.seed(opt.rng_seed)
    torch.set_printoptions(threshold=50, precision=4)

    print('-'*80)
    pprint(vars(opt))

    main(opt)

    print('-'*80)
    pprint(vars(opt))
