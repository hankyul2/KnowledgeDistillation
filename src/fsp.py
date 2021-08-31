import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class FSP(nn.Module):
    def __init__(self, soft_factor=3, kd_lambda=1):
        super(FSP, self).__init__()
        self.kd_lambda = kd_lambda
        self.soft_factor = soft_factor


    def forward(self, act_input_s, act_input_t, act_output_s, act_output_t):
        return torch.stack([F.mse_loss(g_s, g_t.detach()) / self.soft_factor * self.kd_lambda for g_s, g_t in
                          zip(self.fsp_mat(act_input_s, act_output_s), self.fsp_mat(act_input_t, act_output_t))]).mean()


    def fsp_mat(self, act_input, act_output):
        fsp_matrices = []
        for f_in, f_out in zip(act_input, act_output):
            if f_in.size(2) > f_out.size(2):
                f_in = F.adaptive_avg_pool2d(f_in, f_out.shape[2:])
            fsp_matrices.append(rearrange(f_in, 'b c h w -> b c (h w)') @ rearrange(f_out, 'b c h w -> b (h w) c') \
                        / (f_in.size(2) * f_in.size(3)))
        return fsp_matrices

