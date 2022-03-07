from time import time
import torch

n_iteration = int(1e7)

a = torch.rand(50) > .5
b = torch.rand(50) > .5

t1 = time()
for _ in range(n_iteration):
    a & b
print(time() - t1)

a = a.float()
b = b.float()
t1 = time()
for _ in range(n_iteration):
    a + b
print(time() - t1)
