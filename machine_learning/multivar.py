# -*- coding: utf-8 -*-
"""
Created on Fri May  3 06:21:45 2019

@author: nathp
"""

def transpose(X,a,b):
    out=[]
    for i in range (b):
        temp=[]
        for j in range (a):
            temp.append(X[j][i])
        out.append(temp)
    return out
#print(transpose([[1,2,3],[3,4,5]],2,3))
def multiply(X,Y,a,b,c):
    out=[]
    for i in range(a):
        temp=[]
        for j in range(b):
            t=0
            for k in range(c):
               t+=X[i][k]*Y[k][j]
            temp.append(t)
        out.append(temp)
    return out
def linearsolver(A,b):
    n=len(A)
    M=A
    i=0
    for x in M:
        x.append(b[i])
        i+=1
    for k in range(n):
        for i in range(k,n):
            if abs(M[i][k])>abs(M[k][k]):
                M[k],M[i]=M[i],M[k]
            else:
                pass
        for j in range(k+1,n):
            q=float(M[j][k])/M[k][k]
            for m in range(k,n+1):
                M[j][m]-=q*M[k][m]
        x=[0 for i in range(n)]
        x[n-1]=float(M[n-1][n])/M[n-1][n-1]
        for i in range(n-1,-1,-1):
            z=0
            for j in range(i+1,n):
                z+=float(M[i][j])*x[j]
            x[i]=(M[i][n]-z)/M[i][i]
    return x

#print(linearsolver([[2,3],[1,-9]],[5,7]))
x=[]
y=[]
with open("multivar.txt","r") as a:
    for line in a:
        temp=[]
        temp.append(line.split("\t")[0])
        temp.append(line.split("\t")[1])
        y.append(line.split("\t")[2])
        x.append(temp)
print(x,y)