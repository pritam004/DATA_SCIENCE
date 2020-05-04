import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
def normalize(X):
	X=transpose(X)
	for i in range (len(X)):
		s=0
		for j in range(len(X[0])):
	   		s+=X[i][j]**2
		for j in range(len(X[0])):
			X[i][j]=X[i][j]/m.sqrt(m.fabs(s))
	return transpose(X)
def multiply(X,Y):
    w, h = len(X), len(Y[0]);
    res = [[0 for x in range(w)] for y in range(h)]
    #print(res)
    for i in range(len(X)):
       # iterate through columns of Y
       for j in range(len(Y[0])):
           # iterate through rows of Y
           for k in range(len(Y)):
               res[i][j] += X[i][k] * Y[k][j]
    return res
def transpose(X):
    res=[]
    for i in range(len(X[0])):
        temp=[]
        for j in range(len(X)):
            temp.append(X[j][i])
        res.append(temp)
        
    return res
import math as m
def bubbleSort(arr,x):
    n = len(arr)
 
    # Traverse through all array elements
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            x=transpose(x)
            if m.fabs(arr[j]) <m.fabs(arr[j+1]) :
                arr[j], arr[j+1] = arr[j+1], arr[j]
                x[j],x[j+1]=x[j+1],x[j]
            x=transpose(x)
            y=np.array(x)
    return arr,y
def file_read():
	matrix=open('/home/pritam/Documents/programs/lap/assignment2/mnist_train.csv').read()
	matrix=[item.split() for item in matrix.split('\n')[:-1]]
	#print(matrix)
	t=[]
	for i in matrix :
		temp=[j.split(',') for j in i] 
		t.append(temp[0])
	for i in range(len(t)):
		for j in range(len(t[0])):
			t[i][j]=int(t[i][j])
	t=np.array(t)
	
	return t
def test():
	#A=t
	#A=[[5,5,5],[7,4,-6],[4,3,5]]
	#print(A)ValueError: operands could not be broadcast together with shapes (785,10000) (10000,785) 
	#A=[[1,-1],[-2,2],[2,-2]]
	import sys
	'''matrix=open('/home/pritam/Documents/programs/lap/assignment2/mnist_train.csv').read()
	matrix=[item.split() for item in matrix.split('\n')[:-1]]
	for i in range(len(matrix)):
		matrix[i]=[int(j) for j in matrix[i]]
	A=matrix'''
	#print(A)
	#print(np.array(A).shape)
	A=[[4,11,14],[8,7,-2]]
	return A
def svd(A):
	AAt=np.dot(np.array(np.transpose(A)),np.array(A))
	AtA=np.dot(np.array(A),np.array(np.transpose(A)))

	aat=np.array(AAt)
	ata=np.array(AtA)
	eigenValues,eigenVectors=LA.eig(aat)
	#w2,v2=LA.qr(aat)
	#print(v)
	#print(w)
	#print(v2)
	#print(w2)
	eigenValues1,eigenVectors1=LA.eig(ata)
	#print(w1)
	#print(v1)'''
	#print(matrix)
	print(A)
	S= np.zeros((len(A),len(A[0])))
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	idx1 = eigenValues1.argsort()[::-1]   
	eigenValues1 = eigenValues1[idx1]
	eigenVectors1 = eigenVectors1[:,idx1]
	#w,v=bubbleSort(w,v)
	#print(v)
	
	w=eigenValues
	v=eigenVectors
	w1=eigenValues1
	v1=eigenVectors1
	print(v)

	for i in range(len(A)):
	    S[i][i]=m.sqrt(m.fabs(w[i]))
	nv=normalize(v)
	#nv=v
	nvt=transpose(nv)
	#print(nvt)
	nv1=normalize(v1)
	#nv1t=transpose(nv1)
	nv1t=nv1
	S=np.array(S)
	#nvt=nvt.tolist()
	nvt=np.array(nvt)
	print(S)
	#print(S.shape())
	Z=np.dot(S,nvt)
	print(Z)
	print(nv1t)
#	print("this is final output\n" + str(np.dot(np.array(nv1t),Z)))
#	print(S)
#	print(nvt)
#	print(nv1t)
	'''a,b,c=LA.svd(A)
	print("library")
	print(a,b,c)
matrix=open('/home/pritam/Documents/programs/lap/assignment2/listfile.txt').read()
matrix=[item.split() for item in matrix.split('\n')[:-1]]
#print(matrix)
t=[]
for i in matrix :
	temp=[j.split(',') for j in i] 
	t.append(temp[0])
for i in range(len(t)):
	for j in range(len(t[0])):
		t[i][j]=int(t[i][j])
t=np.array(t)
AAt=t
matrix=open('/home/pritam/Documents/programs/lap/assignment2/listfile1.txt').read()
matrix=[item.split() for item in matrix.split('\n')[:-1]]
#print(matrix)
t=[]
for i in matrix :
	temp=[j.split(',') for j in i] 
	t.append(temp[0])
for i in range(len(t)):
	for j in range(len(t[0])):
		t[i][j]=int(t[i][j])
t=np.array(t)

AtA=t'''

def truncated_svd(A,dim,w,v):
	
	
	#AAt=np.dot(np.array(np.transpose(A)),np.array(A))
	#AtA=np.dot(np.array(A),np.array(np.transpose(A)))

	#aat=np.array(AAt)
	#ata=np.array(AtA)
	#w,v=LA.eig(AAt)
	#w=m.fabs(w)
	#w2,v2=LA.qr(aat)
	#print(v)
	#print(w)
	#print(v2)
	#print(w2)
	#w1,v1=LA.eig(AtA)
	
	#print(w)
	#print(w1)
	#np.around(w1,2)
	#print(w1)
	dim=min(dim,len(w))
	#print(w1)
	#print(w1==w)
	#print(v1)
	#print(matrix)
	#print(A)
	S= np.zeros((dim,dim))
	z=np.argsort(w)
	#z=np.filp(z)
	z= z[::-1]
	#z=np.array(w).argpartition(-1*dim)[-1*dim:]
	w = w[z]
	v = v[:,z]
	#print(w)
	#w,v=bubbleSort(w,v)
	#w1,v1=bubbleSort(w1,v1)
	#print(v)
	for i in range(dim):
	    S[i][i]=m.sqrt(m.fabs(w[i]))
	#print(S)
	v=v[:,:dim]
	#v1t=transpose(v1)
	#v1=v1[:,:dim]
	#print(v)
	#print(v1)

	nv=normalize(v)
	nv=np.array(nv)
	u=np.dot(A,v)
	for i in range(len(u[0])):
		for j in range(len(u)):
			u[j][i]/=S[i][i]
	nvt=transpose(nv)
	#print(nvt)
	#nv1=normalize(v1)
	#nv1t=transpose(nv1)
	#nv1t=nv1
	#S=np.array(S)
	Z=np.dot(S,nvt)
	res=np.dot(u,Z)
	print("this is final output\n" + str(res))
	
	#print(S)
	#print(nvt)
	#print(nv1t)
	#a,b,c=LA.svd(A)
	#print("library")
	#print(a,b,c)
	'''from sklearn.decomposition import TruncatedSVD
	from sklearn.random_projection import sparse_random_matrix
	svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
	svd.fit(A)
	print(svd.singular_values_)  '''
	return res

def truncated_lib():
	svd = TruncatedSVD(2)
	X=svd.fit(test())
	print(svd.singular_values_)  

def rmse_n(A,n,w,v):
	B=truncated_svd(A,n,w,v)
	print(B.shape)
	s=0
	for i in range(len(A)):
		for j in range(len(A[0])):
			s+=(A[i][j]-B[i][j])**2
	s=m.sqrt(s)/m.sqrt(10000*784)
	return s
def rmse(A):
	error=[]
	num=[2,5,10,20,50,100,200,500]
	AAt=np.dot(np.array(np.transpose(A)),np.array(A))
	#AtA=np.dot(np.array(A),np.array(np.transpose(A)))

	aat=np.array(AAt)
	#ata=np.array(AtA)
	w,v=LA.eig(AAt)
	for i in num:
		error.append(rmse_n(A,i,w,v))
	plt.plot(num,error)
	plt.show()
#rmse(file_read())
#print(truncated_svd(test(),1))
	
			
		
	
#truncated_svd(file_read(),5)
#svd(test())
