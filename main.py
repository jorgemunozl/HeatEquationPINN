import random as rd
import numpy as np
def mostarm(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j],end=" ")
        print()
def dot(a,b):
    sum=0
    for i in range(len(a)):
        sum+=a[i]*b[i]
    return sum
def sig(x):
    return 1/(1+np.exp(-x))
def dsig(x):
    return (sig(x)**2)*np.exp(-x)

number=int(input("Enter the number of inputs: "))
num_out=int(input("Enter the number of outputs: "))
num_neur=int(input("Enter the number of neurons in each layer: "))
num_layers=int(input("Enter the amount hidden layers: "))
output=[]
input=[rd.randint(1,20) for _ in range(number)]
bias=[]
weights=[]
for i in range(num_layers+1):
    if i==0:
        b=[rd.randint(1,20) for _ in range(num_neur)]
        w=[[rd.randint(1,20) for _ in range(number)] for _ in range(num_neur)]
    elif i==num_layers:
        b=[rd.randint(1,20) for _ in range(num_out)]
        w=[[rd.randint(1,20) for _ in range(num_neur)] for _ in range(num_out)]
    else:
        b=[rd.randint(1,20) for _ in range(num_neur)]
        w=[[rd.randint(1,20) for _ in range(num_neur)] for _ in range(num_neur)]
    weights.append(w)
    bias.append(b)

print("weigths: ")
for i in range(len(weights)):
    print("----- w ",i+1,"-----")
    mostarm(weights[i])
    print("-----")
print("bias: ",bias)

a=[]
a.append(input)
for i in range(len(weights)):
    al=[]
    for j in range(len(bias[i])):
        value=dot(a[i],weights[i][j])    
        al.append(value)
    a.append(al)
print(a)
#output=a[len(a)-1]

#for i in range(4):
    #sum=0
    #for j in range(4):
        #sum+=a2[j]*w3[i][j]
    #a3[i]=sum        

#for i in range(2):
    #sum=0
    #for j in range(len(a3)):
     #   sum+=a3[j]*w4[i][j]
    #a4[i]=sum
#for i in range(2):
    #print("Salida:",i+1,a4[i])