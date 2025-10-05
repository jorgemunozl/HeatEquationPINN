# Train an MLP in PyTorch to learn f(x) = 10*(x - x**2)**2 + 3 on [0, 1]

import math
import random
import torch
import torch.nn as nn

# Reproducibility
torch.manual_seed(42)
random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target function
def initial_condition(x: torch.Tensor) -> torch.Tensor:
    # x: (..., 1)
    return 10.0 * (x - x**2)**2 + 3.0

# Hyperparameters
N_SAMPLES = 4096           # number of training points
HIDDEN = 64                # width of hidden layers
LR = 1e-3                  # learning rate
EPOCHS = 3000              # training epochs
PRINT_EVERY = 300          # logging frequency

# Data (uniform on [0,1])
x_train = torch.rand(N_SAMPLES, 1, device=device)
y_train = initial_condition(x_train)

# Model: simple MLP
model = nn.Sequential(
    nn.Linear(1, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, 1),
).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop (full-batch)
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Train MSE: {loss.item():.6e}")

# Evaluate on a test grid
with torch.no_grad():
    x_test = torch.linspace(0.0, 1.0, 201, device=device).unsqueeze(-1)
    y_true = initial_condition(x_test)
    y_fit = model(x_test)
    test_mse = nn.functional.mse_loss(y_fit, y_true).item()

print(f"\nTest MSE on 201-point grid: {test_mse:.6e}")

# Example: show a few predictions vs truth
with torch.no_grad():
    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xv = torch.tensor([[val]], device=device)
        print(f"x={val:.2f}  f(x)={initial_condition(xv).item():.6f}  pred={model(xv).item():.6f}")
