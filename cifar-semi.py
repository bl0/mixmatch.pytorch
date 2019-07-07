"""Train CIFAR10 with PyTorch."""
import argparse
import os
import random
import time
from functools import partial
from pprint import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.nn.functional import mse_loss, softmax, log_softmax

from lib.dataset import UnlabeledCIFAR10, PseudoCIFAR10
from lib.wideresnet import WideResNet
from lib.utils import AverageMeter, accuracy, guess_label, mixup, interleave, WeightEMA


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
    print('-' * 80)
    print('selected labeled indexes: ', labeled_indexes)

    return test_loader, train_loader, unlabeled_train_loader


best_acc = 0
def build_model(args):
    net = WideResNet(num_classes=args.num_class).to(args.device)
    ema_net = WideResNet(num_classes=args.num_class).to(args.device)
    for param in ema_net.parameters():
        param.detach_()

    print('#param: {}'.format(sum([p.nelement() for p in net.parameters()])))

    if args.device == 'cuda':
        cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(net, ema_net, args, alpha=0.999)

    # resume from pretrained model
    global best_acc
    if len(args.resume) > 0:
        print('==> Resuming from pretrained checkpoint..')
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])
        ema_net.load_state_dict(checkpoint['ema_net'])
        args.start_step = checkpoint['step'] + 1
        best_acc = checkpoint['best_acc']

    return net, ema_net, optimizer, ema_optimizer


def validate(val_loader, model, device='cpu', print_freq=100, prefix='test'):
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(f'{prefix}: [{i}/{len(val_loader)}] '
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

        print(f' * {prefix} Prec@1 {top1.avg:.3f}')

    return top1.avg


# Training
def train(net, ema_net, optimizer, ema_optimizer, trainloader, unlabeled_trainloder, testloader, writer, args):
    end = time.time()

    def inf_generator():
        while True:
            for data in zip(trainloader, unlabeled_trainloder):
                yield data

    for step, ((x_in, l_in), y_in) in enumerate(inf_generator(), start=args.start_step):
        if step >= args.max_iters:
            break
        data_time = time.time() - end

        with torch.no_grad():
            x_in = x_in.to(args.device)
            l_in = l_in.to(args.device)
            y_in = [yi.to(args.device) for yi in y_in]
            guess = guess_label(y_in, net).detach_()

            nu = len(y_in)
            bs = x_in.shape[0]
            assert x_in.shape[0] == y_in[0].shape[0]

            # mixup
            l_in_onehot = torch.zeros(bs, args.num_class).float().to(args.device).scatter_(1, l_in.view(-1, 1), 1)
            xy, l_xy = mixup(torch.cat([x_in] + y_in, dim=0),
                             torch.cat([l_in_onehot] + [guess] * nu, dim=0))
            # reshape to (nu+1, bs, w, h, c)
            xy = xy.reshape([nu + 1] + list(x_in.shape))
            # reshape to (nu+1, bs)
            l_xy = l_xy.reshape([nu + 1] + list(l_in_onehot.shape))
            x, y = xy[0], xy[1:]
            l_x, l_y = l_xy[0], l_xy[1:]

        # forward. only update bn in one step
        net.train()
        batches = interleave([x, y[0], y[1]], bs)
        logits = [net(batches[0])]
        for batchi in batches[1:]:
            logits.append(net(batchi))
        logits = interleave(logits, bs)
        logits_x = logits[0]
        logits_y = torch.cat(logits[1:], 0)

        # logits_x = net(x)
        # logits_y = net(y.reshape([-1, ] + list(x_in.shape)[1:]))

        # loss
        # cross entropy loss for soft label
        loss_xe = torch.mean(torch.sum(-l_x * log_softmax(logits_x, dim=-1), dim=1))
        # L2 loss
        loss_l2u = mse_loss(softmax(logits_y, dim=-1), l_y.reshape(nu * bs, args.num_class))
        # weight for unlabeled loss with warmup
        w_match = 75 * min(1, step / 16384)
        loss = loss_xe + w_match * loss_l2u

        prec1, = accuracy(logits_x, l_in, topk=(1,))

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
            writer.add_scalar('top1/train_ema', get_acc(trainloader, ema_net, prefix='train_ema'), log_step)
            writer.add_scalar('top1/val', get_acc(testloader, net, prefix='val'), log_step)
            val_ema_acc = get_acc(testloader, ema_net, prefix='val_ema')
            writer.add_scalar('top1/val_ema', val_ema_acc, log_step)

            global best_acc
            if val_ema_acc > best_acc:
                best_acc = val_ema_acc
                state = {
                    'step': step,
                    'best_acc': best_acc,
                    'net': net.state_dict(),
                    'ema_net': ema_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                os.makedirs(args.model_dir, exist_ok=True)
                torch.save(state, os.path.join(args.model_dir, 'ckpt.pth.tar'))

            print('best accuracy: {:.2f}\n'.format(best_acc))


def main(args):
    # Data
    testloader, trainloder, unlabeled_trainloader = get_dataloader(args)
    net, ema_net, optimizer, ema_optimizer = build_model(args)

    if args.eval:
        return validate(testloader, net, device=args.device, print_freq=args.print_freq)

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
    parser.add_argument('--start-step', default=0, type=int,
                        metavar='N', help='useful for resume')
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

    print('-' * 80)
    pprint(vars(opt))

    main(opt)

    print('-' * 80)
    pprint(vars(opt))
