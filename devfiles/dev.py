import torch

from deep_morpho.models import BiMoNNClassifierLastLinearNotBinary


x = torch.ones((3, 1, 51, 51))
n_classes = 10

model = BiMoNNClassifierLastLinearNotBinary(
    kernel_size=(7, 7),
    channels=[5],
    atomic_element='bisel',
    input_size=x.shape[1:],
    n_classes=n_classes,
)

model.layers[0:]

otp = model(x)
assert otp.shape == (x.shape[0], n_classes)