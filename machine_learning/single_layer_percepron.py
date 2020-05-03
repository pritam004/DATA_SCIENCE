import matplotlib.pyplot as plt
import math
d=0
c=0
#two features and 2 classes
x=[]
y1=[]
w=[]
#for plotting
p1=[]
p2=[]
p3=[]
p4=[]
def dot_product(X,Y,n):
    d=0
    for i in range(n):
        d+=X[i]*Y[i]
    return d
def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))
  else:
    return 1/(1 + math.exp(-gamma))
#print(str(dot_product([-1,-1],[2,3],2)))
print("enter the no of features")
d=int(input())
print("enter the no of classes")
c=int(input())
with open ('pla_data.csv','r') as a:
    for line in a:
        temp=[1.0]
        temp.append(float(line.split('\t')[0]))
        temp.append(float(line.split('\t')[1]))
        x.append(temp)
        y1.append(float(line.split('\t')[2]))
        if float(line.split('\t')[2]) >0 :
                p1.append(float(line.split('\t')[0]))
                p2.append(float(line.split('\t')[1]))
        else :
                p3.append(float(line.split('\t')[0]))
                p4.append(float(line.split('\t')[1]))
#transform y
y=[]
for i in range(100):
    if y1[i] >0 :
        y.append([1,0])
    else:
        y.append([0,1])
epsilon=.2
temp=[]
u=[]
v=[]
def phi(n):
    d= 1/(1+math.exp(-n))
    return d
def phi_dif(n):
    d=math.exp(-n)/((1+math.exp(-n))*(1+math.exp(-n)))
    return d
e=0        
for i in range(c):
    temp=[]
    for j in range(d+1):
        temp.append(1)
    w.append(temp)
z=0;
while z<1000 :
    #print("------------------------------------------------------------------")
    u=[]
    v=[]
    for n in range(100):
        e=0
        t1=[]
        t2=[]
        
        for j in range(c):
            t1.append(float(dot_product(x[n],w[j],3)))
            t2.append(sigmoid(float(dot_product(x[n],w[j],3))))
        u.append(t1)
        v.append(t2)
        for j in range(c):
            e+=((y[n][j]-v[n][j])*(y[n][j]-v[n][j]))/2
            for i in range(d+1):
                w[j][i]+=float(.1*float(y[n][j]-v[n][j])*phi_dif(float(u[n][j]))*x[n][i])
        
    #print(e)
   # print(v)
    #print(w)
    z+=1
#print(v)
x=[]
y1=[]
with open ('perceptron_test.txt','r') as a:
    for line in a:
        temp=[1.0]
        temp.append(float(line.split('\t')[0]))
        temp.append(float(line.split('\t')[1]))
        x.append(temp)
        y1.append(float(line.split('\t')[2]))
y=[]
for i in range(100):
    if y1[i] >0 :
        y.append([1,0])
    else:
        y.append([0,1])
u=[]
v=[]
for n in range(100):
    e=0
    t1=[]
    t2=[]
        
    for j in range(c):
        t1.append(float(dot_product(x[n],w[j],d+1)))
        t2.append(sigmoid(float(dot_product(x[n],w[j],d+1))))
    u.append(t1)
    v.append(t2)

for i in range(100):
        if v[i][0]>.5:
            print(1)
        else:
            print(-1)
for i in range(100):
        if (v[i][0]>.5)and (y[i][0]>0):
            print(1)
        elif (v[i][0]<.5)and (y[i][0]<0):
            print(-1)
        else:
            print(0)           
print(v2)


