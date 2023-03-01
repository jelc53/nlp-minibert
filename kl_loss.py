import torch.nn.functional as F


def kl_loss_stable(input, target, epsilon=1e-6, reduce=True):
    """ Numerically stable version of kl_loss from authors of SMART
    Source: https://github.com/namisan/mt-dnn/blob/master/mt_dnn/loss.py
    """
    input = input.view(-1, input.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = input.size(0)
    p = F.log_softmax(input, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
    ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
    if reduce:
        return (p * (rp - ry) * 2).sum() / bs
    else:
        return (p * (rp - ry) * 2).sum()


def kl_loss(input, target, reduction='batchmean'):
    """Kullback-Leibler divergence using pytorch in-builts"""
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )


def kl_loss_sym(input, target, reduction='sum', alpha=1.0):
    """Symmetric Kullback-Leibler divergence using pytorch in-builts
    Source: https://arxiv.org/pdf/1911.03437.pdf
    """
    loss = F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )
    return loss * alpha
