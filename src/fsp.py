import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class FSP(nn.Module):
    def __init__(self, factor=4):
        super(FSP, self).__init__()
        self.factor = factor

    def forward(self, act_input_s, act_input_t, act_output_s, act_output_t):
        return torch.cat([self.factor * F.mse_loss(g_s, g_t) for g_s, g_t in
         zip(self.get_g(act_input_s, act_output_s), self.get_g(act_input_t, act_output_t))]).mean()


    def get_g(self, act_input, act_output):
        return [rearrange(f_in, 'b c h w -> b c (h w)') @ rearrange(f_out, 'b c h w -> b (h w) c')
                for f_in, f_out in zip(act_input, act_output)]

