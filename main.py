import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def function_to_optimize(x):
    return (x - 3) ** 2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.fc1= nn.Linear(10,10)
        self.fc2= nn.Linear(10,20)
        self.fc3= nn.Linear(20,1)

    def forward(self,x):
        x= nn.ReLU()(self.fc1(x))
        x= nn.ReLU()(self.fc2(x))
        x= self.fc3(x)
        return x


def main():
    x = torch.linspace(-10, 10, 100)
    y = function_to_optimize(x)
        
    
    plt.plot(x.numpy(), y.numpy())
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()