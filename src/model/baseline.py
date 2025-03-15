import numpy as np
import torch

from torch import nn

from transformers import PreTrainedModel


class AllZerosModel(nn.Module):

    def __init__(self):
        super(AllZerosModel, self).__init__()

    def forward(self, x):
        h, w = x.shape

        return torch.tensor([zero_pred() for _ in range(h * w)]).reshape(h, w, 2)


class AllOnesModel(nn.Module):

    def __init__(self):
        super(AllOnesModel, self).__init__()

    def forward(self, x):
        h, w = x.shape

        return torch.tensor([one_pred() for _ in range(h * w)]).reshape(h, w, 2)


class NormalDistModel(nn.Module):

    def forward(self, x):
        h, w = x.shape
        sample = np.random.rand(h * w)

        return torch.tensor([sample_to_val(it) for it in sample]).reshape(h, w, 2)


class UniformDistModel(nn.Module):

    def forward(self, x):
        h, w = x.shape
        sample = np.random.normal(loc=0.5, scale=0.1667, size=h*w)

        return torch.tensor([sample_to_val(it) for it in sample]).reshape(h, w, 2)


def zero_pred():
    return [1, 0]

def one_pred():
    return [0, 1]

def sample_to_val(x):
    if x < 0.5:
        return zero_pred()
    else:
        return one_pred()
