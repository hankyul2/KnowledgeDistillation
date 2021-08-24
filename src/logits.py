from torch import nn
import torch.nn.functional as F

class Logits(nn.Module):
    def __init__(self):
        super(Logits, self).__init__()

    def forward(self, s_feat, t_feat):
        return F.mse_loss(s_feat, t_feat)