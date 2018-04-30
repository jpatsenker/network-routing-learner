from copy import copy
import numpy as np
import os, sys
sys.path.insert(0, os.getcwd())
from core.user import User
import pickle as pickle
import random
import math

def deg_to_rad(deg):
	return deg * math.pi / 180.0

EARTH_RAD_KM = 6371.0

def sig1(dist, a=500., t=1000.):
	return a * (np.tanh(1./a*(dist - t))+1)

def distance(p0, p1, rad = EARTH_RAD_KM):
	p0r = (deg_to_rad(p0[0]),deg_to_rad(p0[1]))
	p1r = (deg_to_rad(p1[0]),deg_to_rad(p1[1]))
	dlat = abs(p0r[0]-p1r[0])
	dlon = abs(p0r[1]-p1r[1])
	a = math.sin(dlat/2.)**2 + math.cos(p0r[0]) * math.cos(p1r[0]) * math.sin(dlon / 2.)**2
	d_sigma = 2. * math.asin(math.sqrt(a))
	return rad*d_sigma

def bfs(graph,i):
	bfo = [-1] * len(graph)
	q = [i]
	l = i
	count = 0
	while len(q)>0:
		z = q.pop(0)
		bfo[z] = count
		for f in graph[z].friends:
			if bfo[f]==-1:
				bfo[f]=-2
				q.append(f)
		if z==l and len(q)>0:
			count+=1
			l=q[-1]
	return bfo



def theta(x):
	return 1./(1. + np.exp(-x))

def simulation(graph,weights):
	graph_size=len(graph)

	source = int(random.random()*graph_size)
	destination = int(random.random()*graph_size)
	curr = source
	hops = 0
	while curr != destination:
		neighbors = graph[curr].friends
		neighbor_totals = np.zeros(len(neighbors))
		for i in range(len(neighbors)):
			neighbor=neighbors[i]
			n = graph[neighbor]
			deg = n.deg1
			wdeg = math.log(deg)
			d = graph[destination]
			dist = distance(n.pos,d.pos)
			wdist = sig1(dist)
			cic = float(len(set(n.comm).intersection(d.comm)))
			medpower = np.median(list(map(lambda z: graph[z].deg1, n.friends)))
			locality = sum(map(lambda z: distance(graph[z].pos,graph[source].pos), n.friends))
			features = np.array([dist,wdist,cic,deg,wdeg,medpower,locality])
			neighbor_totals[i] = theta(np.dot(weights,features))

		r = random.random()
		s = 0
		i=0
		while s<r:
			s+=neighbor_totals[i]
			i+=1
		curr=neighbors[i]
		hops += 1
	return hops, bfs(graph,d)[s]


gowalla_weights = np.array([1.468081706129472086e-01,-1.545969938352645734e+00,-5.089143668535733855e-01,4.370629297416210868e+00,-5.072072365082430423e+00,-8.362934819850959656e+00,-4.214174338656954122e-01])
airnet_weights = np.array([-0.43524995,-2.07752073,-0.57762063,1.1298535,-1.03288428,-1.65432984,-0.14868317])

with open('data/airport_net/airnet.pkl','rb') as w:
	airnet = pickle.load(w)

with open('GraphSets/test_graph.pkl','rb') as w:
	gowalla = pickle.load(w)


print(simulation(airnet, airnet_weights))