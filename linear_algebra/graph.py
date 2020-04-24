import matplotlib.pyplot as plt
import math as m
k=1
ek=.5/m.sqrt(k)
ekv=[]
while ek>.000615:
	ek=.5/m.sqrt(k)
	ekv.append(ek)
	k+=1


