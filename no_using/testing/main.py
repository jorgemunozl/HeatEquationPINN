import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


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
epochs = 4000
batch_size = 100
x = torch.linspace(-10, 10, 50).unsqueeze(1)
y = function_to_optimize(x).float()

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


save_path = "testing/trained_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs,
}, save_path)

print(f"Saved trained model and optimizer state to {save_path}")

with torch.no_grad():
    predictions = model(x)
    print("Final MSE:", criterion(predictions, y).item())
    plt.plot(x.numpy(), y.numpy(), label="True")
    plt.plot(x.numpy(), predictions.numpy(), label="Predicted")
    plt.legend()
    plt.xlabel("x (normalized)")
    plt.ylabel("y")
    plt.title("Model Predictions vs True Values")
    plt.savefig("testing/testing.png")
