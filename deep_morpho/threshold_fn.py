import torch
from math import pi


def arctan_threshold(x):
    """
    Calculate the arctan threshold

    Args:
        x: write your description
    """
    return 1/pi * torch.arctan(x) + 1/2


def tanh_threshold(x):
    """
    Tanh threshold function

    Args:
        x: write your description
    """
    return 1/2 * torch.tanh(x) + 1/2


def sigmoid_threshold(x):
    """
    Sigmoid threshold function

    Args:
        x: write your description
    """
    return torch.sigmoid(x)


def erf_threshold(x):
    """
    Erf threshold function that accepts input x

    Args:
        x: write your description
    """
    return 1/2 * torch.erf(x) + 1/2


def clamp_threshold(x, s1=0, s2=1):
    """
    Clamp a given threshold.

    Args:
        x: write your description
        s1: write your description
        s2: write your description
    """
    return x.clamp(s1, s2)

