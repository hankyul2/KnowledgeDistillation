from torch import nn
import torch.nn.functional as F


class ST(nn.Module):
    def __init__(self, T=4.0, alpha=1):
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, x_s, x_t):
        return self.alpha * F.kl_div(F.log_softmax(x_s / self.T, dim=1), F.softmax(x_t / self.T, dim=1),
                                     reduction='batchmean') * self.T * self.T
