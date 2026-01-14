import torch
from torch import nn


class GaussianDistribution(nn.Module):
    def __init__(self, moments: torch.Tensor):
        super(GaussianDistribution, self).__init__
        self.mean, self.log_var = torch.chunk(moments, 2, dim=1)
        # self.log_var = torch.sigmoid(self.log_var)*5
        # self.mean = torch.sigmoid(self.mean)

        # Clamp the log of variances
        # self.log_var = torch.clamp(self.log_var, -20, 10)


    def sample(self):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn_like(std)
        return self.mean + eps * std

    def kl(self):
        self.var = torch.exp(self.log_var)
        # a = 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.log_var, dim=[1,2,3,4])
        # a = torch.mean(a)
        b = 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.log_var)
        return b
