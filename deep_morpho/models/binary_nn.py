import torch.nn as nn


class BinaryNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.binary_mode = False

    def binary(self, mode: bool = True):
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
                module.binary(mode)
        return self
