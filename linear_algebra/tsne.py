import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# Importing sklearn and TSNE.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state we define this random state to use this value in TSNE which is a randmized algo.
RS = 25111993

# Importing matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
#%matplotlib inline

# Importing seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
                
                
                
def ts(x):                

	# Loading the vector
	#Data_1 = np.genfromtxt ('mnist_train.csv', delimiter=",")
	# Here we are importing KMeans for clustering Product Vectors
	matrix=open(x).read()
	matrix=[item.split() for item in matrix.split('\n')[:-1]]
	#print(matrix)
	t=[]
	for i in matrix :
		temp=[j.split(',') for j in i] 
		t.append(temp[0])
	for i in range(len(t)):
		for j in range(len(t[0])):
			t[i][j]=int(t[i][j])
	t=np.array(t)
	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=10, random_state=0).fit(t)
	# We can extract labels from k-cluster solution and store is to a list or a vector as per our requirement
	Y=kmeans.labels_ # a vector

	z = pd.DataFrame(Y.tolist()) # a list
	# Fit the model using t-SNE randomized algorithm
	digits_proj = TSNE(random_state=RS).fit_transform(t)
	print(list(range(0,10)))
	sns.palplot(np.array(sns.color_palette("hls", 10)))
	scatter(digits_proj, Y)
	plt.show()





		        
# An user defined function to create scatter plot of vectors
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each cluster.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

            
                
                
                

