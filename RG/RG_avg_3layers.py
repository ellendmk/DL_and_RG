import numpy as np
import os
from random import *
import random
from datetime import datetime
random.seed(datetime.now())

def averageDataRG(data):
    output=np.ones((int(data.shape[0]/2),int(data.shape[0]/2)))
    for i in range(0,data.shape[0]-1,2):
        for j in range(0,data.shape[1]-1,2):
            output[int(i/2),int(j/2)]=(data[i,j]+data[i+1,j]+
                data[i,j+1]+data[i+1,j+1])/4
    return output


outFolder="rg_outputs/"
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

Nv=4096

data_file=["input data filepath"]

data=np.load(data_file)

output=[]
for j in range(len(data)):
    output.append(averageDataRG(data[j].reshape(64,64)).reshape((1024)).copy())

np.savetxt(outFolder + "outputL1_Nv" + str(Nv) + ".csv", output, delimiter=',')

data=output
output2=[]
for j in range(len(output)):
    output2.append(averageDataRG(output[j].reshape(32,32)).reshape((256)).copy())

np.savetxt(outFolder + "outputL2_Nv" + str(Nv) + ".csv", output2,delimiter=',')

output3=[]
for j in range(len(output2)):
    output3.append(averageDataRG(output2[j].reshape(16,16)).reshape((64)).copy())

np.savetxt(outFolder + "outputL3_Nv" + str(Nv) + ".csv", output3, delimiter=',')