import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def act2attn(activation, p=2, norm=False):
    activation = activation.pow(p).mean(dim=1)
    if norm:
        activation = F.normalize(rearrange(activation, 'b h w -> b (h w)'), dim=1)
    return


class AT(nn.Module):
    def __init__(self, p=2, beta=1e+3):
        super(AT, self).__init__()
        self.beta = beta
        self.p = p

    def forward(self, act_s_list, act_t_list):
        loss = [F.mse_loss(act2attn(s, self.p, norm=True), act2attn(t, self.p, norm=True)) for s, t in
                zip(act_s_list, act_t_list)]
        return torch.stack(loss, dim=0).sum(dim=0) * self.beta
