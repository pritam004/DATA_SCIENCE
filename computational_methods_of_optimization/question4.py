import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta =[i for i in range(361)]
'''x1=-3

l1=[]
l2=[]
while(x1<3):
	l1.append(x1)
	l2.append(x1)
	x1+=.01

x3= funct()
'''
x1 = np.linspace(-4, 4, 100)
x2 = np.linspace(-4, 4, 100)
def funct1(x1,x2):
	return (10*x1**2)+(10*x1*x2)+x2**2+4*x1-10*x2+2
def funct2(x1,x2):
	return (16*x1**2)+(8*x1*x2)+10*x2**2+12*x1-6*x2+2


x1, x2 = np.meshgrid(x1, x2)
x3=funct1(x1,x2)
x4=funct2(x1,x2)
ax.plot_surface(x1, x2, x4, cmap=cm.coolwarm,linewidth=0, antialiased=True)
#ax.plot(1.8,-4,funct1(1.8,-4))
#ax.plot(x1,x2,x4)
#ax.plot(1.8,-4,x3)
tr=[]
tr.append(funct2(-.5,.5))
ax.plot([-.5], [.5], tr, markerfacecolor='k', markeredgecolor='k', marker='x', markersize=5, alpha=0.6)
ax.legend()

#plt.show()
ftheta=[]
for i in theta :
	ftheta.append(funct1(1.8+.01*np.cos(i*np.pi/180),-4+.01*np.sin(i*np.pi/180)))


plt.plot(theta,ftheta)	
plt.show()





#func1=[funct(i,j) for i in l1 and j in l2]





