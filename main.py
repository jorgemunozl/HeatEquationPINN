import torch
import torch.nn as nn
import torch.optim as optim


def function_to_optimize(x): #It is important that this function recieve a tensor.
    return (x - 3) ** 2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.Linear(20,1)
        )

    def forward(self, x):
        return self.layers(x)

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)
epochs = 10000
x = torch.linspace(-10,10,100)
y = function_to_optimize(x)

for epoch in range(epochs):
    
    outputs = model(x.unsqueeze(1))
    loss = criterion(outputs, y.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print("loss:",loss.item())


with torch.no_grad():
    preds = torch.argmax(model(x), dim=1)
    acc = (preds == y).float().mean()
    print("Final accuracy")
