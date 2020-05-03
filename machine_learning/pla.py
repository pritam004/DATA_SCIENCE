import matplotlib.pyplot as plt

x=[]
y=[]
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
print(str(dot_product([-1,-1],[2,3],2)))
with open ('pla_data.csv','r') as a:
    for line in a:
        temp=[1.0]
        temp.append(float(line.split('\t')[0]))
        temp.append(float(line.split('\t')[1]))
        x.append(temp)
        y.append(float(line.split('\t')[2]))
        if float(line.split('\t')[2]) >0 :
                p1.append(float(line.split('\t')[0]))
                p2.append(float(line.split('\t')[1]))
        else :
                p3.append(float(line.split('\t')[0]))
                p4.append(float(line.split('\t')[1]))
        
                
        
for i in range(3):
    w.append(1)
temp=1
while temp==1:
    temp=0
    for i in range (100):
        if float(y[i]*dot_product(w,x[i],3))<0 :
            temp=1
            for j in range(3):
                w[j]+=y[i]*x[i][j]
    print(w)
print(w)

plt.plot(p1, p2, 'ro')
plt.plot(p3,p4,'bo')
plt.axis([-7, 7, -7, 7])
x=[i for i in range(-7,+7)]
y=[(i*w[1]+w[0])/-w[2] for i in x]
plt.plot(x,y)
plt.show()
