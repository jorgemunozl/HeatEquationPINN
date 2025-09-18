Saving and loading PyTorch model parameters

1) Save a full checkpoint (model + optimizer + metadata)

```python
# after training
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs,
}, 'trained_model.pth')
```

Load the full checkpoint for resuming training or inference:

```python
checkpoint = torch.load('trained_model.pth', map_location=torch.device('cpu'))
model = NeuralNetwork()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# If you need to resume training
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint.get('epochs', 0)
```

2) Save only the model parameters (recommended for inference):

```python
# after training
torch.save(model.state_dict(), 'model_state.pth')
```

Load only the model parameters for inference:

```python
state = torch.load('model_state.pth', map_location=torch.device('cpu'))
model = NeuralNetwork()
model.load_state_dict(state)
model.eval()
```

Notes:
- Use `map_location=torch.device('cpu')` if you trained on GPU but run inference on CPU.
- Prefer saving `state_dict()` for portability and smaller files when optimizer state isn't needed.
