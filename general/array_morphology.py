import torch
import torch.nn.functional as F


def format_for_conv(ar, device):
    """
    Format ar as a numpy array for convolution.

    Args:
        ar: write your description
        device: write your description
    """
    return torch.tensor(ar).unsqueeze(0).unsqueeze(0).float().to(device)


def array_erosion(ar, selem, device="cpu"):
    """
    Return True if the given array is a different size than the given array.

    Args:
        ar: write your description
        selem: write your description
        device: write your description
    """
    conv_fn = {2: F.conv2d, 3: F.conv3d}[ar.ndim]

    return (conv_fn(
        format_for_conv(ar, device=device), format_for_conv(selem, device=device),
        padding=selem.shape[0] // 2,
    ) == selem.sum()).squeeze().to("cpu").int().numpy()


def array_dilation(ar, selem, device="cpu"):
    """
    Dilation of an array with a scalar element.

    Args:
        ar: write your description
        selem: write your description
        device: write your description
    """
    conv_fn = {2: F.conv2d, 3: F.conv3d}[ar.ndim]

    return (conv_fn(
        format_for_conv(ar, device=device), format_for_conv(selem, device=device),
        padding=selem.shape[0] // 2,
    ) > 0).squeeze().to("cpu").int().numpy()
