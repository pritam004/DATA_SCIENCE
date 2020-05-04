import numpy as np
import math as m
import networkx as nx
def linearsolver(A,b):
  n = len(A)
  M = A

  i = 0
  for x in M:
   x.append(b[i])
   i += 1

  for k in range(n):
   for i in range(k,n):
     if abs(M[i][k]) > abs(M[k][k]):
        M[k], M[i] = M[i],M[k]
     else:
        pass

   for j in range(k+1,n):
       q = float(M[j][k]) / M[k][k]
       for m in range(k, n+1):
          M[j][m] -=  q * M[k][m]

  x = [0 for i in range(n)]

  x[n-1] =float(M[n-1][n])/M[n-1][n-1]
  for i in range (n-1,-1,-1):
    z = 0
    for j in range(i+1,n):
        z = z  + float(M[i][j])*x[j]
    x[i] = float(M[i][n] - z)/M[i][i]

  return x

	
	

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
def dot(a,b):
	s=0
	for i in range(len(a)):
		s+=a[i]*b[i]
	return s
def spc_dot(a,b):
	s=0
	for i in range(len(a)):
		s+=a[i][0]*b[0][i]
	return s
def norm(a):
	s=0
	for i in range(len(a)):	
		s+=a[i]*a[i]
	s=m.sqrt(s)
	return s
		
def transpose(X):
    res=[]
    for i in range(len(X[0])):
        temp=[]
        for j in range(len(X)):
            temp.append(X[j][i])
        res.append(temp)
        
    return res
def  eye(m):
	temp=[]
	for i in range(m):
		t=[]
		for j in range(m):
			if(i==j):
				t.append(1.0)
			else:
				t.append(0.0)
		temp.append(t)
	temp=np.array(temp)
	return temp
def checkDiagonal(arr):
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			if i == j:
				continue
			else:
				if abs(arr[i][j]) > 0.001:
					return False
	return True
def check_upper_triangular(arr):
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			if i <= j:
				continue
			else:
				if abs(arr[i][j]) > 0.001:
					return False
	return True
def qr1(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder1(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A
 
def make_householder1(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H
def qrFactorization(arr):
	temp = arr
	i = 0
	p=np.eye(len(temp[0]))
	while(True):
		#print(p)
	
		Q,R = qr1(temp)
		temp = np.dot(R, Q)
		p=np.dot(p,Q)
		#print("Q is "+str(Q))
		#print("R is"+ str(R))
		if(check_upper_triangular(temp)):
			print("Number of Factorizations: " + str(i+1))
			break
		else:
			i += 1
	#print(p)
	return temp
def eigvec_from_val(A: np.ndarray, eigval: float) -> np.ndarray:
   
    eigeye = eigval*np.eye(A.shape[-1])
    x = np.random.random((A.shape[-1], 1))  # random vector
    x /= norm(x)  # scale to length 1
    while(True):
        Ax = (np.linalg.inv(A - eigeye)) @ x
        x_old = x
        x = Ax/(np.linalg.norm(Ax)+1e-16)
        if norm(np.abs(x) - np.abs(x_old)) < 1e-8:
            y=[]
            for i in range(len(x)):
                y.append(x[i][0])
            return y
#z=np.diag(qrFactorization([[2,2],[3,3]])).round(4)
#print(z)
'''
for i in range(len(z)):
	print(magic(eigvec_from_val(a,z[i])))'''
#print(qrFactorization(np.array([[-1,2],[3,-4]])))
#print(b)
