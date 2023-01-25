from abc import ABC
import torch.nn as nn


class BinaryNN(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.binary_mode = False

    def binary(self, mode: bool = True, *args, **kwargs):
        r"""Sets the module in binary mode.

        Args:
            mode (bool): whether to set binary mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.binary_mode = mode
        for module in self.children():
            if isinstance(module, BinaryNN):
                module.binary(mode, *args, **kwargs)
        return self

    def forward_save(self, x):
        return {'output': self.forward(x)}

    def numel_binary(self):
        res = self._specific_numel_binary()
        for module in self.children():
            if isinstance(module, BinaryNN):
                res += module.numel_binary()
        return res

    def numel_float(self):
        return sum([param.numel() for param in self.parameters() if param.requires_grad])

    def _specific_numel_binary(self):
        """Specifies the number of binarizable parameters that are not contained in the children."""
        return 0

    # def to(self, device, *args, **kwargs):
    #     super().to(device=device, *args, **kwargs)
    #     self.device = device

    # def cuda(self, device, *args, **kwargs):
    #     super().cuda(device=device, *args, **kwargs)
    #     self.device = device
    @property
    def device(self):
        for param in self.parameters():
            return param.device
