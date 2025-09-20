import torch


x_col = torch.empty(3).uniform_(-10.0, 10.0)
x_col = x_col.view(3, 1)

print(x_col)
print(x_col.dim())
