import random as rd
import numpy as np
def mostarm(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(round(matrix[i][j],1),end=" ")
        print()
def dot(a,b):
    return np.dot(a,b)
def sig(x):
    return 1/(1+np.exp(-x))
def dsig(x):
    return (sig(x)**2)*np.exp(-x)

number_eval=4
number=1
num_out=1
num_neur=30
num_layers=1
input=np.array([number_eval])
weights=[]
bias=[]
for i in range(num_layers+1):
    if i==0:
        n_in=number*num_neur
        b=np.zeros(num_neur)
        w=np.random.rand(num_neur,number)* np.sqrt(2 / n_in)   
    elif i==num_layers:
        n_in=num_neur*num_out
        b=np.zeros(num_out)
        w=np.random.rand(num_out,num_neur   )* np.sqrt(2 / n_in)
    else:
        n_in=num_neur*num_neur
        b=np.zeros(num_neur)
        w=np.random.rand(num_neur,num_neur)* np.sqrt(2 / n_in)
    weights.append(w)
    bias.append(b)

np.savez("bias.npz", *bias) 
np.savez("weights.npz",*weights)
#for i in range(len(weights)):
#    print("----- w ",i+1,"-----")
#    mostarm(weights[i])
#    print("-----")
#print("bias: ",bias)