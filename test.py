import torch

x = torch.arange(1,28).reshape(3,3,3)
y = x.reshape(9,3)
z = y.reshape(3,3,3)
print(x)
print(y)
print(z)