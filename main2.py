import torch

x = torch.linspace(-10,10,100)
print(x)
print(x.shape)
print(x.unsqueeze(1).shape)