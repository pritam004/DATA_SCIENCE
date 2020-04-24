import math as m
def zoof1 (x):
	return abs(2-x)+abs(5-4*x)+abs(8-9*x)
	
def zoof2 (x):
	return 3*(x-1)**2-m.exp(x-1)

def fibo_list(N):
	fib=[]
	for i in range (N):
		if i ==0 or i==1:
			fib.append(1)
		else :
			fib.append(fib[i-1]+fib[i-2])
		
	return fib	
	
print(fibo_list(10))
def gsr(a,b,zoof,eps):
	z=0
	xl=a
	xu=b
	l=.5*(1+m.sqrt(5))
	tolerance=.5*(xu-xl)
	while(tolerance>eps):
		d=(xu-xl)*l/(1+l)
		xm=xu-d 
		xp=xl+d
		if(zoof==1):
			if(zoof1(xm)<zoof1(xp)):
				xu=xp
				z+=2
			else:
				xl=xm
				z+=2
		else:
			if(zoof2(xm)<zoof2(xp)):
				xu=xp
				z+=2
			else:
				xl=xm
				z+=2
		tolerance=.5*(xu-xl)
	return .5*(xl+xu),tolerance,z

def fs(a,b,zoof,N):
	
	z=0
	xl=a
	xu=b
	f=[]
	f.append(1)
	f.append(0)
	tolerance=.5*(xu-xl)
	fibo=fibo_list(N)
	for i in range(2,N+1):
		ro=fibo[N-i]/fibo[N-i+1]
		d=(xu-xl)*ro
		xm=xu-d 
		xp=xl+d
		if(zoof==1):
			if(zoof1(xm)<zoof1(xp)):
				xu=xp
				z+=2
			else:
				xl=xm
				z+=2
		else:
			if(zoof2(xm)<zoof2(xp)):
				xu=xp
				z+=2
			else:
				xl=xm
				z+=2
		tolerance=.5*(xu-xl)
	return .5*(xl+xu),tolerance,z


def problem():
	a=0
	b=3
	zoof=1
	N=5
	print("\nRunning for N="+str(N)+"and zoof"+str(zoof))
	x,tol,count=fs(a,b,zoof,N)
	x1,tol1,count1=gsr(a,b,zoof,tol)
	print("fibonacci search returns \n x="+str(x)+"\ntolerance="+str(tol)+"\n count="+str(count))
	print("\ngolden search returns \n x="+str(x1)+"\ntolerance="+str(tol1)+"\n count="+str(count1))
	zoof=1
	N=10
	print("\nRunning for N="+str(N)+"and zoof"+str(zoof))
	x,tol,count=fs(a,b,zoof,N)
	x1,tol1,count1=gsr(a,b,zoof,tol)
	print("fibonacci search returns \n x="+str(x)+"\ntolerance="+str(tol)+"\n count="+str(count))
	print("\ngolden search returns \n x="+str(x1)+"\ntolerance="+str(tol1)+"\n count="+str(count1))
	zoof=1
	N=20
	print("\nRunning for N="+str(N)+"and zoof"+str(zoof))
	x,tol,count=fs(a,b,zoof,N)
	x1,tol1,count1=gsr(a,b,zoof,tol)
	print("fibonacci search returns \n x="+str(x)+"\ntolerance="+str(tol)+"\n count="+str(count))
	print("\ngolden search returns \n x="+str(x1)+"\ntolerance="+str(tol1)+"\n count="+str(count1))
	zoof=2
	N=5
	print("\nRunning for N="+str(N)+"and zoof"+str(zoof))
	x,tol,count=fs(a,b,zoof,N)
	x1,tol1,count1=gsr(a,b,zoof,tol)
	print("fibonacci search returns \n x="+str(x)+"\ntolerance="+str(tol)+"\n count="+str(count))
	print("\ngolden search returns \n x="+str(x1)+"\ntolerance="+str(tol1)+"\n count="+str(count1))
	zoof=2
	N=10
	print("\nRunning for N="+str(N)+"and zoof"+str(zoof))
	x,tol,count=fs(a,b,zoof,N)
	x1,tol1,count1=gsr(a,b,zoof,tol)
	print("fibonacci search returns \n x="+str(x)+"\ntolerance="+str(tol)+"\n count="+str(count))
	print("\ngolden search returns \n x="+str(x1)+"\ntolerance="+str(tol1)+"\n count="+str(count1))
	zoof=2
	N=20
	print("\nRunning for N="+str(N)+"and zoof"+str(zoof))
	x,tol,count=fs(a,b,zoof,N)
	x1,tol1,count1=gsr(a,b,zoof,tol)
	print("fibonacci search returns \n x="+str(x)+"\ntolerance="+str(tol)+"\n count="+str(count))
	print("\ngolden search returns \n x="+str(x1)+"\ntolerance="+str(tol1)+"\n count="+str(count1))
	interval=[]
	z1=[]
	z2=[]
	i=0
	while i<=3 :
		interval.append(i)
		i+=.1
	for i in interval:
		z1.append(zoof1(i))
		z2.append(zoof2(i))
	m.plot(interval,z1)



problem()		

#print(zoof2(5))
