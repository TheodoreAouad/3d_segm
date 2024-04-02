import numpy as np
import torch

from .preprocessing import Preprocessor



class MinMaxPrepocess(Preprocessor):
    def __call__(self, img: np.ndarray):
        return (img - img.min()) / (img.max() - img.min())


class MaskedMinMaxNormChannels(Preprocessor):
    def __init__(self, background: float = 0, channels=None):
        self.background = background
        self.channels = channels

    def __call__(self, tensor):
        if self.background is None:
            return self.min_max_norm(tensor)
        res = tensor + 0

        if hasattr(self, 'channels') and self.channels is not None:
            iterator = set(range(tensor.shape[0])).intersection(self.channels)
        else:
            iterator = range(tensor.shape[0])

        if tensor.squeeze().ndim == 3:
            for chan in iterator:
                fg = tensor[chan][tensor[chan] != self.background]
                if len(fg.unique()) == 1:
                    res[chan] = 1
                else:
                    res[chan][res[chan] != self.background] = self.min_max_norm(fg)
        else:
            fg = tensor[tensor != self.background]
            res[res != self.background] = self.min_max_norm(fg)

        return res

    def min_max_norm(self, tensor: torch.Tensor) -> torch.Tensor:

        if tensor.squeeze().ndim == 3:
            mins = torch.zeros(tensor.shape[0]).to(tensor.device)
            maxs = torch.zeros(tensor.shape[0]).to(tensor.device)
            for chan in range(tensor.shape[0]):
                mins[chan], _ = tensor[chan].min()
                maxs[chan], _ = tensor[chan].max()


        else:
            mins = tensor.min()
            maxs = tensor.max()
        return tensor.sub_(mins).div_(maxs - mins)

    def __repr__(self):
        return self.__class__.__name__ + '()'