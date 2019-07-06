import time

import torch

from lib.utils import AverageMeter, accuracy


def validate(val_loader, model, device='cpu', print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
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
                print(f'Test: [{i}/{len(val_loader)}] '
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

        print(f' * Prec@1 {top1.avg:.3f}')

    return top1.avg
