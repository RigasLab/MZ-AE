import torch

x = torch.arange(1,37).reshape(3,3,4)
x = torch.movedim(x,-1,1)
y = torch.flatten(x, start_dim = 0, end_dim = 1)
c = torch.arange(1,13).reshape(4,3)
z = y.reshape(x.shape)
# z = y.reshape(3,3,3)
print(x)
print(y.shape)
print(z)
print(z.shape)
print(c.shape)
print((3,*c.shape))