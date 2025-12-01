import torch
import torch.nn.functional as F


def calc_reg_loss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


def contrast_loss(embeds1, embeds2, indices, temp):
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    pckEmbeds1 = embeds1[indices]
    pckEmbeds2 = embeds2[indices]
    nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -torch.log(nume / deno).mean()

