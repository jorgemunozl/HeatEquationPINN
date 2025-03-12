import numpy as np

def sig(x):
    return np.maximum(0, x)


def dsig(x):
    return np.where(x > 0, 1, 0) 

input=np.array([])
data=np.loadtxt("funcion_valores.csv",delimiter=",")
weights_l=np.load("weights.npz")
weights=[]

weights.append(weights_l["arr_0"])
weights.append(weights_l["arr_1"])
bias_l=np.load("bias.npz")
bias=[]
bias.append(bias_l["arr_0"].reshape(30,1))
bias.append(bias_l["arr_1"].reshape(1,1))
#print(weights[0].shape)
#print(weights[1].shape)
#print(bias[0].shape)
#print(bias[1].shape)
costo=0
learning_rate=0.0001
print("comenzar calculo")
for i in range(1000):
    input=np.array([data[i][0]])
    #print(input)
    input=input.reshape(1,1)
    a=[input,None,None]
    for j in range(2):
        a[j+1]=sig(np.matmul(a[j].T,weights[j].T)+bias[j].T)
        a[j+1]=a[j+1].T
        #print("a",a[j])
        #print("weights",weights[j])
        #print("bias",bias[j])
        #Z = np.matmul(a[j].T, weights[j].T) + bias[j].T
        #print(a[j+1])
    delta=[None,None]
    #print("a",a)
    deriva_z=dsig(a[-1])
    #print("primero")
    print(deriva_z)
    delta[1]=(a[-1]-data[i][1])*deriva_z
    #print("segundo")
    #print(delta[1])
    arg=np.matmul(weights[1].T,delta[1])
    delta[0]=arg*dsig((np.matmul(a[0].T,weights[0].T)+bias[0].T).T)
    #print(delta)
    #print("inittttt")
    #print(weights[1])
    weights[1]=weights[1]-learning_rate*(np.matmul(delta[1],a[1].T))
    #print("tercero")
    #print(delta[1])
    #print("cuarto")
    #print((np.matmul(delta[1],a[1].T)))
    #print("enddddd")
    #print(weights[1])
    weights[0]=weights[0]-learning_rate*(np.matmul(delta[0],a[0]))
    bias[1]=bias[1]-learning_rate*delta[1]
    bias[0]=bias[0]-learning_rate*delta[0]
    valor=(a[-1]-data[i][1])**2
    costo+=valor
costo=costo/2000
print("costo: ",costo)
np.savez("bias_t.npz", *bias)
np.savez("weights_t.npz",*weights)
print("terminar calculo")

