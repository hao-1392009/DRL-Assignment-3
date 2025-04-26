import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0):
        super().__init__()
        c = in_features ** -0.5
        self.w_mu = nn.Parameter(torch.rand(out_features, in_features) * 2*c - c)
        self.w_sigma = nn.Parameter(torch.full((out_features, in_features), sigma0 * c))

        self.b_mu = nn.Parameter(torch.rand(out_features) * 2*c - c)
        self.b_sigma = nn.Parameter(torch.full((out_features,), sigma0 * c))

        self.w_epsilon = None
        self.b_epsilon = None

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return F.linear(
            x,
            self.w_mu + self.w_sigma * self.w_epsilon,
            self.b_mu + self.b_sigma * self.b_epsilon
        )

    def resample_noise(self):
        self.b_epsilon = self._f(torch.randn(self.out_features))
        self.w_epsilon = torch.outer(self.b_epsilon, self._f(torch.randn(self.in_features)))

    def _f(self, x: torch.Tensor):
        return x.sign() * x.abs().sqrt()
