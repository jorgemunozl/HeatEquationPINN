import os
import torch
import torch.nn as nn
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


CHECKPOINT_PATH = "trained_model.pth"
loaded = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
model = NeuralNetwork()
state = loaded.get('model_state_dict', loaded)
model.load_state_dict(state)
model.eval()

x = np.linspace(-5, 15, 100)
y = function_to_optimize(x)

X = torch.from_numpy(x.astype(np.float32)).unsqueeze(-1)
Y = []

with torch.no_grad():
    preds = model(X).squeeze(-1).cpu().numpy()
    Y = preds

mse = np.mean((Y - y) ** 2)
print(f"Inference MSE on [10,20]: {mse:.6f}")

plt.figure()
plt.plot(x, y, label="True")
plt.plot(x, Y, label="Predicted")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Inference: True vs Predicted")
plt.grid(True)
plt.savefig("prove.png")
print("Saved plot to prove.png")
