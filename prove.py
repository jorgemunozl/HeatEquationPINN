import torch
import torch.nn.functional as F

y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.5, 1.7, 3.2])

loss = F.mse_loss(y_pred, y_true)
print(loss)  # a scalar tensor
