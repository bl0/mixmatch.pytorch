"""Train CIFAR10 with PyTorch."""
import argparse
import os
import random
import time
from functools import partial
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from lib.datasets.cifar import UnlabeledCIFAR10, PseudoCIFAR10
from lib.models import WideResNet, resnet18_cifar
from lib.utils import accuracy,  cross_entropy, guess_label, mixup, interleave, WeightEMA
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

    get_loader = partial(DataLoader, batch_size=args.batch_size, num_workers=args.num_workers)

    ds = PseudoCIFAR10(labeled_indexes=labeled_indexes, root=args.data_dir, transform=transform_train)
    train_loader = get_loader(ds, drop_last=True, shuffle=True)
    
    ds_u = UnlabeledCIFAR10(indexes=unlabeled_indexes, nu=2, root=args.data_dir, transform=transform_train)
    unlabeled_train_loader = get_loader(ds_u, drop_last=True, shuffle=True)
    
    testset = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = get_loader(testset, shuffle=False)
    print('-'*80)
    print('selected labeled indexes: ', labeled_indexes)

    return test_loader, train_loader, unlabeled_train_loader


def build_model(args):
    def create_model(ema=False):
        net = WideResNet(num_classes=args.num_class).to(args.device)
        if ema:
            for param in net.parameters():
                param.detach_()
        return net

    net = create_model(ema=False)
    ema_net = create_model(ema=True)

    print('#param: {}'.format(sum([p.nelement() for p in net.parameters()])))

    if args.device == 'cuda':
        cudnn.benchmark = True

    # resume from unsupervised pretrain
    if len(args.resume) > 0:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['net'].items()
                                if k in model_dict
                                and v.size() == model_dict[k].size()}
        assert len(pretrained_dict) > 0
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    return net, ema_net

# Training
def train(net, ema_net, optimizer, ema_optimizer, trainloader, unlabeled_trainloder, testloader, writer, args):
    best_acc = 0
    end = time.time()
    def inf_generator(trainloader, unlabeled_trainloder):
        while True:
            for data in zip(trainloader, unlabeled_trainloder):
                yield data

    for step, ((x_in, l_in), y_in) in enumerate(inf_generator(trainloader, unlabeled_trainloder)):
        if step >= args.max_iters:
            break
        data_time = time.time() - end

        with torch.no_grad():
            x_in = x_in.to(args.device)
            l_in = l_in.to(args.device)
            y_in = [yi.to(args.device) for yi in y_in]
            guess = guess_label(y_in, net).detach()

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

        # forward. only update bn in one step
        # switch to train mode
        net.train()
        batches = interleave([x, y[0], y[1]], bs)
        logits = [net(batches[0])]
        # switch to eval mode: no update bn
        # net.eval()
        for batchi in batches[1:]:
            logits.append(net(batchi))
        logits = interleave(logits, bs)

        logits_x = logits[0]
        logits_y = torch.cat(logits[1:], 0)

        # loss
        loss_xe = cross_entropy(input=logits_x, target=l_x)
        loss_l2u = nn.functional.mse_loss(nn.functional.softmax(logits_y, dim=-1), l_y.reshape(nu * bs, args.num_class))
        w_match = 75 * min(1, step / 16384)  # w_match with warmup
        loss = loss_xe + w_match * loss_l2u

        prec1, = accuracy(logits_x, l_in)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        log_step = args.batch_size * step
        if step % args.print_freq == 0:
            writer.add_scalar('w_match', w_match, log_step)
            writer.add_scalar('top1/train', prec1.item(), log_step)
            writer.add_scalar('loss/all', loss.item(), log_step)
            writer.add_scalar('loss/xe', loss_xe.item(), log_step)
            writer.add_scalar('loss/l2u', loss_l2u.item(), log_step)

            print(f'Train: [{step}/{args.max_iters}] '
                  f'Time: {batch_time:.3f}  '
                  f'Data: {data_time:.3f} '
                  f'prec1: {prec1.item():.3f} '
                  f'w_match: {w_match:.3f} '
                  f'Loss: {loss.item():.3f} '
                  f'Loss_xe: {loss_xe.item():.3f} '
                  f'Loss_l2u: {loss_l2u.item():.4f}')

        if (step + 1) % args.eval_freq == 0 or step == args.max_iters - 1:
            ema_optimizer.step(bn=True)
            get_acc = partial(validate, device=args.device, print_freq=args.print_freq)
            acc = get_acc(testloader, net)
            writer.add_scalar('top1/val', acc, log_step)
            writer.add_scalar('top1/val_ema', get_acc(testloader, ema_net), log_step)
            writer.add_scalar('top1/train_ema', get_acc(trainloader, ema_net), log_step)

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
    testloader, trainloder, unlabeled_trainloader = get_dataloader(args)

    print('==> Building model..')
    net, ema_net = build_model(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(net, ema_net, args, alpha=0.999)

    if args.eval:
        return validate(testloader, net,
                        device=args.device, print_freq=args.print_freq)
    # summary writer
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    train(net, ema_net, optimizer, ema_optimizer,
          trainloder, unlabeled_trainloader, testloader, writer, args)


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
    parser.add_argument('--lr', default=0.002, type=float,
                        metavar='LR', help='learning rate')
    parser.add_argument('--resume', '-r', default='', type=str,
                        metavar='FILE', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='test only')
    parser.add_argument('--finetune', action='store_true',
                        help='only training last fc layer')
    parser.add_argument('-j', '--num-workers', default=2, type=int,
                        metavar='N', help='number of workers to load data')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='batch size')
    parser.add_argument('--max-iters', default=500000, type=int,
                        metavar='N', help='number of iterations')
    parser.add_argument('--num-labeled', default=250, type=int,
                        metavar='N', help='number of labeled data')
    parser.add_argument('--rng-seed', default=0, type=int,
                        metavar='N', help='random number generator seed')
    parser.add_argument('--gpus', default='5', type=str, metavar='GPUS')
    parser.add_argument('--eval-freq', default=1024, type=int,
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
