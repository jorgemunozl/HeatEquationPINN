import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Dirichlet Conditions.


"""
def residual_heat_equation(x,t):
    return

def boundary_condition(x,t):
    return

def initial_condition(x):
    return np.arctan(x)*np.sin(x)
"""


def function_to_optimize(x):
    return (x+3)**2-10*x + np.sin(x)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 40),
            nn.Tanh(),
            nn.Linear(40, 80),
            nn.Tanh(),
            nn.Linear(80, 1)
        )

    def forward(self, x):
        return self.layers(x)


model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
epochs = 10000
batch_size = 10
x = torch.linspace(-10, 10, 100).unsqueeze(1)
y = function_to_optimize(x)

for epoch in range(epochs):
    permutation = torch.randperm(x.size(0))
    epoch_loss = 0
    for i in range(0, x.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        x_batch, y_batch = x[indices], y[indices]
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/x.size(0):.4f}")


with torch.no_grad():
    predictions = model(x)
    print("Final MSE:", criterion(predictions, y).item())
    plt.plot(x.numpy(), y.numpy(), label="True")
    plt.plot(x.numpy(), predictions.numpy(), label="Predicted")
    plt.legend()
    plt.xlabel("x (normalized)")
    plt.ylabel("y")
    plt.title("Model Predictions vs True Values")
    plt.savefig("model_predictions.png")
