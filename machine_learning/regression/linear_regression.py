# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:56:48 2019

@author: nathp
"""
x=[]
y=[]
r=0
with open("linear_regression.csv","r") as a:
    for line in a :
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
        r+=1
x2s=0
xs=0
ys=0
xys=0
xb=0
yb=0
xyb=0
x2b=0
for i in range(r):
    xs+=x[i]
    ys+=y[1]
    xys+=x[i]*y[i]
    x2s+=x[i]*x[i]
xb=xs/r
yb=ys/r
xyb=xys/r
x2b=x2s/r

r=(xyb-xb*yb)/(x2b-xb*xb)
print(r)
    
print(x)
print(y)