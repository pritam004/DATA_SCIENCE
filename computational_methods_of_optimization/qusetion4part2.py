import matplotlib.pyplot as plt
import numpy as np
def funct1(x1,x2):
	return (10*x1**2)+(10*x1*x2)+x2**2+4*x1-10*x2+2
def funct2(x1,x2):
	return (16*x1**2)+(8*x1*x2)+10*x2**2+12*x1-6*x2+2
theta =[]
i=0
while i< 2*np.pi:
	theta.append(i)
	i+=.1
ftheta1=[]
for i in theta :
	ftheta1.append(funct1(1.8+.01*np.cos(i),-4+.01*np.sin(i)))
ftheta2=[]
for i in theta :
	ftheta2.append(funct2(-.5+.01*np.cos(i),.5+.01*np.sin(i)))

ftheta11=[]
for i in ftheta1 :
	ftheta11.append(i-funct1(1.8,-4))
ftheta22=[]
for i in ftheta2 :
	ftheta22.append(i-funct2(-.5,.5))

plt.plot(theta,ftheta22)
#fig.suptitle('test title', fontsize=20)
plt.xlabel('theta', fontsize=18)
plt.ylabel('f(x-ad)-f(x)', fontsize=16)	
plt.show()

