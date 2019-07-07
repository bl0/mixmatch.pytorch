import torch

from lib.wideresnet import WideResNet


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


def guess_label(ys, net, T=0.5):
    logits = net(torch.cat(ys, dim=0))

    p = torch.nn.functional.softmax(
        logits, dim=-1).view(len(ys), -1, logits.shape[1])
    p_target = p.mean(dim=0).pow(1. / T)
    p_target /= p_target.sum(dim=1, keepdim=True)

    return p_target


def mixup(x, l, beta=0.75):
    assert x.shape[0] == l.shape[0]
    mix = torch.distributions.Beta(beta, beta).sample(
        (x.shape[0], )).to(x.device).view(-1, 1, 1, 1)

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


class WeightEMA(object):
    def __init__(self, model, ema_model, args, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = WideResNet(num_classes=10).cuda()  # TODO
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)
