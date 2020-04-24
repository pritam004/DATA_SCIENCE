import networkx as nx 
from heapq import heappush, heappop
from itertools import count
#H = nx.read_gml('lclassico.gml')
def _single_source_shortest_path_basic(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]
    while Q:   # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:   # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma

def _single_source_dijkstra_path_basic(G, s, weight):
    # modified from Eppstein
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []   # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + edgedata.get(weight, 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma
def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _accumulate_endpoints(betweenness, S, P, sigma, s):
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return betweenness


def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _rescale(betweenness, n, normalized,
             directed=False, k=None, endpoints=False):
    if normalized:
        if endpoints:
            if n < 2:
                scale = None  # no normalization
            else:
                # Scale factor should include endpoint nodes
                scale = 1 / (n * (n - 1))
        elif n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def _rescale_e(betweenness, n, normalized, directed=False, k=None):
    if normalized:
        if n <= 1:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / (n * (n - 1))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

def degree_centrality(G, nodes): 
	top = set(nodes) 
	bottom = set(G) - top 
	s = 1.0/len(bottom) 
	centrality = dict((n,d*s) for n,d in G.degree_iter(top)) 
	s = 1.0/len(top) 
	centrality.update(dict((n,d*s) for n,d in G.degree_iter(bottom))) 
	return centrality 
def closeness_centrality(G, u=None, distance=None, normalized=True): 
	
	if distance is not None: 

		# use Dijkstra's algorithm with specified attribute as edge weight 
		path_length = functools.partial(nx.single_source_dijkstra_path_length, 
										weight=distance) 
	else: 
		path_length = nx.single_source_shortest_path_length 

	if u is None: 
		nodes = G.nodes() 
	else: 
		nodes = [u] 
	closeness_centrality = {} 
	for n in nodes: 
		sp = path_length(G,n) 
		totsp = sum(sp.values()) 
		if totsp > 0.0 and len(G) > 1: 
			closeness_centrality[n] = (len(sp)-1.0) / totsp 

			# normalize to number of nodes-1 in connected part 
			if normalized: 
				s = (len(sp)-1.0) / ( len(G) - 1 ) 
				closeness_centrality[n] *= s 
		else: 
			closeness_centrality[n] = 0.0
	if u is not None: 
		return closeness_centrality[u] 
	else: 
		return closeness_centrality 
def betweenness_centrality(G, k=None, normalized=True, weight=None, 
						endpoints=False, seed=None): 
	betweenness = dict.fromkeys(G, 0.0) # b[v]=0 for v in G 
	if k is None: 
		nodes = G 
	else: 
		random.seed(seed) 
		nodes = random.sample(G.nodes(), k) 
	for s in nodes: 

		# single source shortest paths 
		if weight is None: # use BFS 
			S, P, sigma = _single_source_shortest_path_basic(G, s) 
		else: # use Dijkstra's algorithm 
			S, P, sigma = _single_source_dijkstra_path_basic(G, s, weight) 

		# accumulation 
		if endpoints: 
			betweenness = _accumulate_endpoints(betweenness, S, P, sigma, s) 
		else: 
			betweenness = _accumulate_basic(betweenness, S, P, sigma, s) 

	# rescaling 
	betweenness = _rescale(betweenness, len(G), normalized=normalized, 
						directed=G.is_directed(), k=k) 
	return betweenness 
def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, 
						weight='weight'): 
	
	from math import sqrt 
	if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph: 
		raise nx.NetworkXException("Not defined for multigraphs.") 

	if len(G) == 0: 
		raise nx.NetworkXException("Empty graph.") 

	if nstart is None: 

		# choose starting vector with entries of 1/len(G) 
		x = dict([(n,1.0/len(G)) for n in G]) 
	else: 
		x = nstart 

	# normalize starting vector 
	s = 1.0/sum(x.values()) 
	for k in x: 
		x[k] *= s 
	nnodes = G.number_of_nodes() 

	# make up to max_iter iterations 
	for i in range(max_iter): 
		xlast = x 
		x = dict.fromkeys(xlast, 0) 

		# do the multiplication y^T = x^T A 
		for n in x: 
			for nbr in G[n]: 
				x[nbr] += xlast[n] * G[n][nbr].get(weight, 1) 

		# normalize vector 
		try: 
			s = 1.0/sqrt(sum(v**2 for v in x.values())) 

		# this should never be zero? 
		except ZeroDivisionError: 
			s = 1.0
		for n in x: 
			x[n] *= s 

		# check convergence 
		err = sum([abs(x[n]-xlast[n]) for n in x]) 
		if err < nnodes*tol: 
			return x 

	raise nx.NetworkXError("""eigenvector_centrality(): 
power iteration failed to converge in %d iterations."%(i+1))""") 
def degree_centrality(G):
  
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.degree()}
    return centrality

'''print("the eigenvector entrality is \n"+ str(eigenvector_centrality(H)))
print("the betweenness entrality is \n"+ str(betweenness_centrality(H)))
print("the closeness entrality is \n"+ str(closeness_centrality(H)))
print("the degree entrality is \n"+ str(degree_centrality(H)))'''

