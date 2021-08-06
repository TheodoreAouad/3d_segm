import torch
import torch.nn as nn
import torch.nn.functional as F

class Unfolder:

    def __init__(self, kernel_size, padding=0):
        self.kernel_size = kernel_size
        self.padding = (padding, padding, padding, padding) if not isinstance(padding, tuple) else padding

        self._right_operator = {}
        self._left_operator = {}
        # self.right_operator = self.create_right_operator(self.size, self.kernel_size[1])
        # self.left_operator = self.create_right_operator(self.size[::-1], self.kernel_size[0]).T

    def unfold(self, x):
        x = F.pad(x, self.padding)
        return self.get_left_operator(x.shape[1:]) @ x @ self.get_right_operator(x.shape[1:])

    @staticmethod
    def create_right_operator(size, k):
        right_operator = torch.zeros((size[0], k * (size[1] - k+1)))
        for i in range(right_operator.shape[0] - k + 1):
            right_operator[i:i+k, k*i:k*(i+1)] = torch.eye(k)
        return right_operator

    def get_right_operator(self, size):
        if size not in self._right_operator.keys():
            self._right_operator[size] = self.create_right_operator(size, self.kernel_size[1])
        return self._right_operator[size]

    def get_left_operator(self, size):
        if size not in self._left_operator.keys():
            self._left_operator[size] = self.create_right_operator(size[::-1], self.kernel_size[0]).T
        return self._left_operator[size]

    def __call__(self, x):
        return self.unfold(x)


class DilationLayerSum(nn.Module):

    def __init__(self, kernel_size, padding: int = 0, init_value: float = -10):
        super().__init__()
        self.kernel_size = kernel_size
        self.init_value = init_value
        self.padding = (padding, padding, padding, padding) if not isinstance(padding, tuple) else padding

        self.unfolder = Unfolder(kernel_size=kernel_size, padding=padding)
        self.weight = nn.Parameter(self.init_weights(init_value)).float()
        self.maxpool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        output = self.unfolder(x)
        output = output + self.kron_weight
        return self.maxpool(output)

    def init_weights(self, init_value):
        weights = torch.zeros(self.kernel_size) + init_value
        weights[self.kernel_size[0]//2, self.kernel_size[1]//2] = 0
        return weights

    @property
    def kron_weight(self):
        return torch.kron(torch.ones([k + 2*self.padding[0] for k in self.kernel_size]), self.weight)
