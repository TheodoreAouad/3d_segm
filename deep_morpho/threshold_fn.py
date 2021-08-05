import torch
from math import pi


def arctan_threshold(x, P):
    return 1/pi * torch.arctan(P * x) + 1/2


def tanh_threshold(x, P):
    return 1/2 * torch.tanh(P * x) + 1/2


def sigmoid_threshold(x, P):
    return torch.sigmoid(P * x)


def erf_threshold(x, P):
    return 1/2 * torch.erf(P * x) + 1/2
