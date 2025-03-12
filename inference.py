import numpy as np
import matplotlib.pyplot as plt
def sig(x):
    return np.maximum(0, x)


def dsig(x):
    return np.where(x > 0, 1, 0) 

inputes=np.array([])
data=np.loadtxt("funcion_valores.csv",delimiter=",")
weights_l=np.load("weights_t.npz")
weights=[]

weights.append(weights_l["arr_0"])
weights.append(weights_l["arr_1"])
bias_l=np.load("bias_t.npz")
bias=[]
bias.append(bias_l["arr_0"].reshape(30,1))
bias.append(bias_l["arr_1"].reshape(1,1))

#print(weights)
#print(bias)

costo=0
print("comenzar calculo")
for i in range(1000):
    inputes=np.array([data[i][0]])
    inputes=inputes.reshape(1,1)
    a=[inputes,None,None]
    for j in range(2):
        a[j+1]=sig(np.matmul(a[j].T,weights[j].T)+bias[j].T)
        a[j+1]=a[j+1].T
    valor=(a[-1]-data[i][1])**2
    costo+=valor
costo=costo/2000
print("costo: ",costo)
print("terminar calculo")
data0=np.linspace(0.5,100,100)
ydata0=np.log(data0)+0.01*data0
data1=[]
ydata1=[]
for i in range(100):
    a=[np.array([i]).reshape(1,1),None,None]
    data1.append(np.array([i]))
    for j in range(2):
        a[j+1]=sig(np.matmul(a[j].T,weights[j].T)+bias[j].T)
        a[j+1]=a[j+1].T
    ydata1.append(a[-1][0][0])
    #print(a[-1][0][0],ydata0[i])
plt.plot(data0,ydata0)
plt.plot(data1,ydata1)

plt.savefig("output.png")  # Guarda la imagen en un archivo