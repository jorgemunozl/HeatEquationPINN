import torch
import matplotlib.pyplot as plt
import numpy as np

# fake "true" data
x = torch.linspace(-2, 2, 100).unsqueeze(1)
y_true = x**3 + 0.1*torch.randn_like(x)

# simple net
import torch.nn as nn
net = nn.Sequential(nn.Linear(1, 50), nn.ReLU(), nn.Linear(50,1))
opt = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

plt.ion()
fig, ax = plt.subplots()

for epoch in range(500):
    opt.zero_grad()
    y_pred = net(x)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    opt.step()
    
    # plot predictions vs true
    ax.clear()
    ax.scatter(x.numpy(), y_true.numpy(), label="True data", alpha=0.5)
    ax.plot(x.numpy(), y_pred.detach().numpy(), 'r-', label="Prediction")
    ax.legend()
    ax.set_title(f"Epoch {epoch}, Loss={loss.item():.4f}")
    
    plt.pause(0.2)

plt.ioff()
plt.show()
