import numpy as np

weights_t=np.load("weights_t.npz")
weightst=[]

weightst.append(weights_t["arr_0"])
weightst.append(weights_t["arr_1"])
bias_t=np.load("bias_t.npz")
biast=[]
biast.append(bias_t["arr_0"].reshape(30,1))
biast.append(bias_t["arr_1"].reshape(1,1))




weights_l=np.load("weights.npz")
weights=[]

weights.append(weights_l["arr_0"][0])
weights.append(weights_l["arr_0"][1].T)
bias_l=np.load("bias.npz")
bias=[]
bias.append(bias_l["arr_0"].reshape(30,1))
bias.append(bias_l["arr_1"].reshape(1,1))

print(weights_t["arr_0"])
print(weights_l["arr_0"][0])
print(weights_t["arr_1"])
print(weights_l["arr_0"][1].T)