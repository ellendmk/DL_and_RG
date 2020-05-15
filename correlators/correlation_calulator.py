import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show, close
from matplotlib import colors
import math
import numpy as np
import os

# display lattice configuration
def displayConfigs(num_configs,data,L=0):
	cmap = colors.ListedColormap([ 'black','white'])
	bounds=[-1,0,1]
	labels=[None]*num_configs
	norm = colors.BoundaryNorm(bounds, cmap.N)
	if num_configs==1:
		plt.imshow(data,interpolation='nearest',cmap=cmap,norm=norm)
	else:

		fig, axes = plt.subplots(nrows=int(np.sqrt(num_configs)), ncols=int(np.sqrt(num_configs)))
		i=0

		for row in axes:
			for col in row:
				if L!=0:
					col.imshow(data[i].reshape([L,L]),cmap=cmap,norm=norm)

					col.set_title(str(i))
				else:
					col.imshow(data[i],cmap=cmap,norm=norm)
					col.set_title(str(label_))
				i=i+1

		fig.subplots_adjust(right=0.8)
	plt.show()

# calculate two point correlator betwen positions x1 and x2
def twoPointFunc(samples,x1,x2,samples2=0):
	corr=0
	if samples2==0:
		a_list=getListAtCoord(samples,x1)
		b_list=getListAtCoord(samples,x2)
		for i in range(0,len(a_list)):
			corr+=float(a_list[i])*b_list[i]
		corr=float(corr)/len(a_list)
		if abs(corr)<0.00001:
			return 0.0
	else:
		a_list=getListAtCoord(samples,x1)
		b_list=getListAtCoord(samples2,x2)
		for i in range(0,len(a_list)):
			corr+=a_list[i]*b_list[i]
		corr=float(corr)/len(a_list)
		if abs(corr)<0.00001:
			return 0.0
	return corr

# Given samples and a location x, calculate <s> for s at location x
def onePointFunc(samples,x):
	corr=0
	a_list=getListAtCoord(samples,x)
	for i in range(0,len(a_list)):
		corr+=a_list[i]

	return float(corr)/len(a_list)

# determine energy operator
def getEpsilon(config, x11, simple=1):
	x12=[-1,-1]
	x21=[-1,-1]
	x10=[-1,-1]
	x01=[-1,-1]
	# |   |x10|
	# |x01|x11|x21|
	# |   |x12|

	x12[0]=x11[0]
	x12[1]=(x11[1]+1)%LENGTH_LATT
	x21[0]=(x11[0]+1)%LENGTH_LATT
	x21[1]=x11[1]

	x01[0]=(x11[0]-1)%LENGTH_LATT
	x01[1]=x11[1]

	x10[0]=x11[0]
	x10[1]=(x11[1]-1)%LENGTH_LATT
	eps=0
	if simple==1:		
		eps+=config[x11[0],x11[1]]*config[x21[0],x21[1]]

	else:
		eps+=config[x11[0],x11[1]]*\
		(config[x12[0],x12[1]]+config[x21[0],x21[1]]+\
		config[x01[0],x01[1]]+config[x10[0],x10[1]])
	return eps

# get epsilon for entire dataset
def getEnergyEpsilonMatrix(samples,simple=1):
	eps_matrix_list=[]
	# |   |x10|
	# |x01|x11|x21|
	# |   |x12|
	x01=[-1,-1]
	x10=[-1,-1]
	x12=[-1,-1]
	x21=[-1,-1]
	mean = 0
	for conf in samples:
		eps_mat=np.ones([LENGTH_LATT,LENGTH_LATT])
		for r in range(0,LENGTH_LATT):
			for c in range(0,LENGTH_LATT):
				eps_mat[r,c]=getEpsilon(conf,[r,c],simple)
		mean += np..mean(eps_mat)
		eps_matrix_list.append(eps_mat)
	mean=mean/len(samples)

	for i in range(0,len(eps_matrix_list)):
		eps_matrix_list[i]-=mean
		
	return eps_matrix_list

#Get value of spin at position x in each sample
def getListAtCoord(samples,x):
	vals_list=[]
	for conf in samples:
		vals_list.append(conf[x[0],x[1]].copy())
	return vals_list

#Get two random coordinates a distance dist apart
def getTwoCoords(dist):
	hor=int(uniform(0,10)>=5)
	x1=[-1,-1]
	x2=[-1,-1]
	if hor==1:
		x1[0]=int(uniform(0,LENGTH_LATT-1))
		x1[1]=int(uniform(0,(LENGTH_LATT-1-dist)%LENGTH_LATT))
		x2[0]=x1[0]
		x2[1]=(x1[1]+dist)%LENGTH_LATT
	else:
		x1[0]=int(uniform(0,(LENGTH_LATT-1-dist)%LENGTH_LATT))
		x1[1]=int(uniform(0,LENGTH_LATT-1))
		x2[1]=x1[1]
		x2[0]=(x1[0]+dist)%LENGTH_LATT
	return [x1,x2]

#Get random coordinate
def getOneCoord():
	x1=[-1,-1]
	x1[0]=int(uniform(0,LENGTH_LATT-1))
	x1[1]=int(uniform(0,LENGTH_LATT-1))
	return x1

def getMeanMat(samples,L):
	meanMat=np.zeros((L,L))
	mean=0
	for config in samples:
		for i in range(0,config.shape[0]):
			for j in range(0,config.shape[1]):
				meanMat[i,j]+=config[i,j]
				mean+=config[i,j]
	mean=mean/(len(samples)*L*L)
	meanMat=meanMat*1.0/len(samples)
	return meanMat

def getTwoPointFuncVals(samples,samples2=0):
	distances=[]
	correlators=[]
	x1=[0,0]
	x2=[0,0]
	if samples2==0:
		for r1 in range(0,LENGTH_LATT):
			for c1 in range(0,LENGTH_LATT):
				for r2 in range(r1,LENGTH_LATT):
					for c2 in range(0,LENGTH_LATT):
						if (r1==r2 and c2>c1) or (r2>r1):
							x1=[r1,c1]
							x2=[r2,c2]
							
							if (math.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)<LENGTH_LATT/2.0):
								distances.append(math.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2))
								correlators.append((twoPointFunc(samples,x1,x2)))
	else:
		for r1 in range(0,LENGTH_LATT):
			for c1 in range(0,LENGTH_LATT):
				for r2 in range(r1,LENGTH_LATT):
					for c2 in range(0,LENGTH_LATT):
						if (r1==r2 and c2>c1) or (r2>r1 and c2!=c1):
							x1=[r1,c1]
							x2=[r1,c2]
							if (math.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)<LENGTH_LATT/2.0):
								distances.append(math.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2))
								correlators.append((twoPointFunc(samples,x1,x2,samples2)))
	return [distances,correlators]

def flows_mag_delta():
	flows = [[] for i in range(60)]
	modelF = "path to supervised net"
	model= models.load_model(modelF)
	
    spinF=''
    for gibbs_updates in range(1,30):
        flows = [[] for i in range(60)]

        print("Gibbs updates = "+str(gibbs_updates))
        folder  = 'path to spins/spins'+str(gibbs_updates)+'_'+str(nsteps)+tempFile+'.npy'
        mfolder = 'path to store magnetisation/'
        
        Xnew = np.load(folder)
        ynew = model.predict_classes(Xnew)
        
        for i in range(0,len(ynew)):
            flows[ynew[i]].append(Xnew[i].copy())

        #calculate average m
        m = [0.0]*60
        for k in range(0,len(flows)):
            print("Flows at T = "+str(k)+" : "+str(len(flows[k])))
            for i in range(0,len(flows[k])):
                mag = np.sum(flows[k][i])
                m[k] += np.abs(mag) / LENGTH_LATT**2
            if len(flows[k]) > 0:
                    m[k] = m[k]/len(flows[k])

        np.savetxt(mfolder + 'm' + str(gibbs_updates) + '.csv', m, delimiter=',')

def flows_2pt_delta():
	flows = [[] for i in range(60)]
	modelF="path to supervised model"
	model= models.load_model(modelF)

	spinsF=''
	
    for gibbs_updates in range(1,30,1):
        flows = [[] for i in range(60)]

        print("gibbs updates = "+str(gibbs_updates))

        folder='path to spins/spins'+str(gibbs_updates)+'_'+str(nsteps)+tempFile+'.npy'
        
        mfolder='corrs/'
        
        if not(os.path.isdir(mfolder)):
            os.mkdir(mfolder)

        Xnew = np.load(folder)
        ynew = model.predict_classes(Xnew)

        for i in range(0,len(ynew)):
            flows[ynew[i]].append(Xnew[i].copy().reshape((10,10)))

        for k in range(0,60):
            if len(flows[k]) > 0:
                [distances1,correlators1] = getTwoPointFuncVals(flows[k])

                mylist = list(sorted(set(distances1)))

                extraCorrs = [0]*len(mylist)
                total      = [0]*len(mylist)
                for i in range(0, len(mylist)):
                    for d in range(0, len(distances1)):
                        if mylist[i] == distances1[d]:
                            extraCorrs[i] += correlators1[d]
                            total[i] += 1

                for i in range(0,len(mylist)):
                    extraCorrs[i] = extraCorrs[i] / total[i]

                np.savetxt(mfolder + 'corr_T_' + str(k) + 'flows_' + str(gibbs_updates)+ '.csv', extraCorrs, delimiter=',')
                np.savetxt(mfolder + 'dist_T_' + str(k) + 'flows_' + str(gibbs_updates) + '.csv',mylist,delimiter=',')

def flows_2pt_deltaEPS():
	flows = [[] for i in range(60)]
	modelF="path_to_supervised_net_model"
	model= models.load_model(modelF)

	spinsF=''

	for rbm_num in range(startnet,startnet+1):
		for gibbs_updates in range(1,30,1):
			flows = [[] for i in range(60)]

			print("Gibbs updates = "+str(gibbs_updates))

			folder='path to spins/spins'+str(gibbs_updates)+'_'+str(nsteps)+tempFile+'.npy'
			mfolder='path to where mags stored/'
			if not(os.path.isdir(mfolder)):
				os.mkdir(mfolder)

			Xnew=np.load(folder)
			ynew = model.predict_classes(Xnew)
			
            for i in range(0,len(ynew)):
				flows[ynew[i]].append(Xnew[i].copy().reshape((10,10)))

			for k in range(1,30):
				if len(flows[k])>1:
					eps_mat = getEnergyEpsilonMatrix(flows[k])
					[distances1,correlators1] = getTwoPointFuncVals(eps_mat)

					mylist = list(sorted(set(distances1)))

					extraCorrs = [0]*len(mylist)
					total      = [0] * len(mylist)

					for i in range(0, len(mylist)):
						for d in range(0, len(distances1)):
							if mylist[i] == distances1[d]:
								extraCorrs[i] += correlators1[d]
								total[i] += 1

					for i in range(0,len(mylist)):
						extraCorrs[i] = extraCorrs[i] / total[i]

					np.savetxt(mfolder + 'corr_T_EPS_'+str(k) + 'flows_' + str(gibbs_updates) + '.csv', extraCorrs, delimiter=',')
					np.savetxt(mfolder + 'dist_T_EPS_' + str(k) + 'flows_' + str(gibbs_updates)+ '.csv', mylist, delimiter=',')


#### MAIN CODE ####
nsteps="10000"
LENGTH_LATT=10

flows_mag_delta()
flows_2pt_delta()
flows_2pt_deltaEPS()
