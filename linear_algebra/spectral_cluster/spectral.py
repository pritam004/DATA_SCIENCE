import networkx as nx
import numpy as np
import math as m
import eigenvalue as e
#import eigenvalue as eig
import matplotlib.pyplot as plt
def spec(H):
	#H = nx.read_gml('lclassico.gml')
	Adj=nx.adjacency_matrix(H)
	arr=nx.to_numpy_matrix(H)
	D = np.diag(np.sum(np.array(Adj.todense()), axis=1))
	arr=D-Adj
	arr=arr.tolist()
	arr=np.array(arr)
	val=np.diag(e.qrFactorization(arr)).round(4)
	z=e.eigvec_from_val(arr,sorted(val)[1])
	for i in range(len(z)):
		if z[i]<0:
			z[i]=-1
		else:
			z[i]=1
	print(z)
	pos=nx.spectral_layout(H	)
	nx.draw_networkx_nodes(H, pos, cmap=plt.get_cmap('jet'),node_color =z , node_size = 100)
	plt.show()
