import sys
def rank_eche(A):	
	rank=0
	for i in range(len(A)):
		flag=0
		for j in range (len(A[0])):
			if A[i][j]!=0:
				#print(A[i][j])
				flag+=1;
		if(flag!=0):
			rank+=1
			#print(rank)
	return rank
def find_pivot_columns(A):
	c=len(A[0])
	r=len(A)
	mat=[]
	for i in range (c):
		mat.append(0)
	for i in range (r):
		for j in range (c):
			if(A[i][j]!=0):
				mat[j]=1
				break
	return mat

#print(find_pivot_columns([[1,2,3],[0,0,5],[0,0,0]]))	
		
#print(rank_eche([[1,2],[3,3]]))

Z=[[1,2,3],[2,4,6],[7,8,9]]
B=[[1,3,1],[5,10,2],[1,8,9]]

def dot_product(A,B):
	n=len(A)
	result=0
	for i in reversed(range (n)):
		result+=(A[i]*B[i])
	return result
def exchange(X,i,j):
	X[i],X[j]=X[j],X[i]
def substract_lists(a,b):
	return [x-y for x,y in zip(a,b)]
def identity(m,n):
	I=[]
	for i in range (m):
		temp=[]
		for j in range (n):
			if(i==j):
				temp.append(1)
			else :
				temp.append(0)
		I.append(temp)
	return I
			
				


ele=[]

def elimination(A):
	r=len(A)
	c=len(A[0])
	pivot=0
	if(r<c):
		rank=r
	else:
		rank=c
	i=0
	while i<r:
	#for i in range(r):
		iden=identity(r,c)
		#print("i is "+str(i)+" pivot is "+str(pivot))
		if A[i][pivot]!=0 :
		#	print("non zero pivot \n\n\n\n")
			for j in range(i+1,r):
				iden=identity(r,c)
				fact= A[j][pivot]/A[i][pivot]
				#print("fact"+str(fact))
				l=[x*fact for x in A[i]]
				l1=[x*fact for x in iden[i]]
				#print("l is " +str(l)+"Aj is "+str(A[j]))
				A[j]=substract_lists(A[j],l);
				
				iden[j]=substract_lists(iden[j],l1)
		#		print("A is "+str(A))
				
		#		print("I is"+ str(iden))
				if(i!=r-1 and iden!=identity(r,c)):
					ele.append(iden)
			if(pivot<c-1):
				pivot+=1
		else:
		#	print("zero pivot\n\n\n\n")
		
			temp=0
			for j in range(i+1,r):
				if(A[j][pivot]!=0):
					temp=j
					break
			if temp==0:
				pivot+=1
				rank-=1
				i-=1
			else:
		#		print("hi"+str(i))
				exchange(A,i,j)
				exchange(iden,i,j)
				i-=1
		#		print("hie"+str(i))
				if iden!=identity(r,c):
					ele.append(iden)
		#	print("A is "+str(A))
			#print("I is "+str(iden))
			#print("i at last is"+str(i))
			
		if(pivot>min(r,c)-1):
			break
		i+=1
	return A,ele,rank

def row_reduced(A):
	r=len(A)
	c=len(A[0])
	pivot=0
	if(r<c):
		rank=r
	else:
		rank=c
	i=0
	while i<r:
	#for i in range(r):
		iden=identity(r,c)
		if A[i][pivot]!=0 :
			#print("non zero pivot \n\n\n\n")
			for j in range(i+1,r):
				iden=identity(r,c)
				fact= A[j][pivot]/A[i][pivot]
			#	print("fact"+str(fact))
				l=[x*fact for x in A[i]]
				l1=[x*fact for x in iden[i]]
			#	print(l)
				A[j]=substract_lists(A[j],l);
				iden[j]=substract_lists(iden[j],l1)
				#print("A is "+str(A))
			fact=1/A[i][pivot]
			for j in range(c):
				 A[i][j]*=fact
			#print(A)
			for j in range(i):
				#print("here")
				
				fact= A[j][pivot]/A[i][pivot]
			#	print("fact"+str(fact)+"i is ",str(i))
				l=[x*fact for x in A[i]]
			#	print(l)
				A[j]=substract_lists(A[j],l);
				
				#print("A is "+str(A))
				
			
			pivot+=1
		else:
			#print("zero pivot\n\n\n\n")
			temp=0
			for j in range(i+1,r):
				if(A[j][pivot]!=0):
					temp=j
					break
			if temp==0:
				pivot+=1
				rank-=1;
				i-=1
			else:
				exchange(A,i,j)
				exchange(iden,i,j)
				i-=1
				#print("i am appending here also")
				if iden!=identity(r,c):
					ele.append(iden)
			#print("A is "+str(A))
			#print("I is "+str(iden))
		
		if(pivot>=min(r,c)):
			break
		i+=1
	return A
def row_reduced_old(A):
	r=len(A)
	c=len(A[0])
	if(c<r):
		pivot=c-1
	else:
		pivot=r-1
	fact=1
	for i in reversed(range(r)):
		print(str(i)+str(pivot)+"ok")
		if(A[i][pivot]!=0):
			while(A[i][pivot-1]!=0 and i!=0):
				pivot-=1
			
			fact=1/A[i][pivot]
			for j in range(c):
				 A[i][j]*=fact
			for j in reversed(range(i)):
				fact=A[j][pivot]/A[i][pivot]
				l=[x*fact for x in A[i]]	
				A[j]=substract_lists(A[j],l);
			pivot-=1
			print(str(A[i][pivot])+str(i)+str(pivot))
	return A
	
	

			
#print(substract_lists([4,5,6],[1,2,3]))
#elimination()
#print("ele is "+str(ele))
#print("A is "+str(A))

def problem1():

	matrix = open(sys.argv[2]).read()
	matrix = [item.split() for item in matrix.split('\n')[:-1]]
	for i in range(len(matrix)):
		matrix[i]= [int(j) for j in matrix[i]]
	A,ele,rank=elimination(matrix)
	rank=rank_eche(A)	
	print("RANK OF THE MATRIX: "+str(rank))
	print("ROW ECHELON FORM:\n"+str(A))
	print("SEQUENCE OF ELEMENTARY MATRICES USED:\n")
	for i in range(len(ele)):
		print(ele[i])
	

def problem2():
	matrix=open(sys.argv[2]).read()
	matrix=[item.split() for item in matrix.split('\n')[:-1]]
	for i in range(len(matrix)):
		matrix[i]=[int(j) for j in matrix[i]]
	#print("matrix is  "+str(matrix))
	A,ele,rank=elimination(matrix)
	B=[]
	#print("matrix is  "+str(matrix))
	for i in range (len (matrix)):
		temp=[]
		for j in range (len(matrix[0])-1):
			temp.append(A[i][j])
		B.append(temp)
	#print("B is "+str(B))
	#print(str(A)+str(B))
	B,ele,rank1=elimination(B)
	rank1=rank_eche(B)
	rank=rank_eche(A)
	#print("A is "+str(A))
	#print("rank is "+str(rank))
	#print("B is "+str(B))
	#print("rank1 is "+str(rank1))
	if(rank1==len(B)):
		print("UNIQUE SOLUTION EXISTS !\n")
	#	print(A)
		S=row_reduced(A)
	#	print(S)

		print("the solution is:")
		for i in range(len(matrix[0])-1):
			print("x"+str(i+1)+"="+str(S[i][len(matrix[0])-1]))
	elif (rank==rank1):
		print ("MANY SOLUTIONS EXIST !\n")
		S=row_reduced(A)
		#print(S)
		c=len(matrix[0])-1
		#print (c)
		S1=row_reduced(B)
		#print("S1 is "+str(S1))
		
		''' old thing----
		infi=[0 for i in range(rank1)]
		#print("initial infi"+str(infi))
		for i in range(c-rank1):
		#	print(i)
			infi.append(1)
		#print(infi)
		for i in reversed(range (rank1)):
			infi[i]=S[i][c]-dot_product(infi,S1[i])
		#	print(infi)
		print("one solution is:")
		for i in range(c):
			print("x"+str(i+1)+"="+str(infi[i]))
		'''
		mat=find_pivot_columns(S1)
		print(S1)
		infi=[0 for i in range(c)]
		for i in range(c):
			if(mat[i]==0):
				infi[i]=1
		print(infi)
		for i in reversed(range(len(matrix))):
			temp=0
			for j in range(c):
				if(S[i][j]!=0):
					temp+=1
					infi[j]=S[i][c]-dot_product(infi,S1[i])
					break
		print("one solution is:")
		for i in range(c):
			print("x"+str(i+1)+"="+str(infi[i]))
					
	else:
		print("NO SOLUTION EXISTS !")
		



if(sys.argv[1]=='problem1'):
	problem1()
if(sys.argv[1]=='problem2'):
	problem2()
#K=[[1,0,2,3],[0,0,1,2],[0,0,0,0]]
#k1=[[1,2,3,7],[0,-8,-18,-47],[0,0,0,0]]

#print(row_reduced(k1))


