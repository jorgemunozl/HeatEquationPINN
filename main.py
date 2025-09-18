import torch
import torch.nn as nn
import torch.optim as optim


def function_to_optimize(x): #It is important that this function recieve a tensor.
    return (x - 3)**2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,20),
            nn.Tanh(),
            nn.Linear(20,1)
        )

    def forward(self, x):
        return self.layers(x)

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
epochs = 100
batch_size = 10
x = torch.linspace(-10,10,100).unsqueeze(1)
x = (x - x.min()) / (x.max() - x.min())  # Normalize to [0,1]
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


import matplotlib.pyplot as plt

with torch.no_grad():
    predictions = model(x)
    print("Final MSE:", criterion(predictions, y).item())
    plt.plot(x.numpy(), y.numpy(), label="True")
    plt.plot(x.numpy(), predictions.numpy(), label="Predicted")
    plt.legend()
    plt.xlabel("x (normalized)")
    plt.ylabel("y")
    plt.title("Model Predictions vs True Values")
    plt.savefig( "model_predictions.png")

"""
with torch.no_grad():
    preds = torch.argmax(model(x), dim=1)
    acc = (preds == y).float().mean()
    print("Final accuracy")
"""
