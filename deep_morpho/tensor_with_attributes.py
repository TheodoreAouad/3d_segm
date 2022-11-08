from torch import Tensor


class TensorGray(Tensor):
    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        if hasattr(self, "gray_values"):
            res.gray_values = self.gray_values.to(*args, **kwargs)
        if hasattr(self, "indexes"):
            res.indexes = self.indexes
        if hasattr(self, "original"):
            res.original = self.original.to(*args, **kwargs)
        return res

    def cuda(self, *args, **kwargs):
        res = super().cuda(*args, **kwargs)
        if hasattr(self, "gray_values"):
            res.gray_values = self.gray_values.to(*args, **kwargs)
        if hasattr(self, "indexes"):
            res.indexes = self.indexes
        if hasattr(self, "original"):
            res.original = self.original.to(*args, **kwargs)
        return res
