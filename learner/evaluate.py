from copy import copy
import numpy as np
import os, sys
sys.path.insert(0, os.getcwd())
from core.user import User
import pickle as pickle
import random
import math
from multiprocessing import Process, Manager

top,bottom=np.array([2.00092774e+04,   1.00000000e+03,   1.00000000e+00, 9.15000000e+02,   6.81892407e+00,   9.15000000e+02, 1.23771376e+07]), np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.])


def reindex_dict(nodes):
	inodes = dict()
	ndic = dict(zip(range(len(nodes.keys())), nodes.keys()))
	rev = {v: k for k, v in ndic.items()}
	for n in range(len(nodes)):
		inodes[n] = copy(nodes[ndic[n]])
		inodes[n].uid = n
	#   if n % 1000 == 0:
	#       print "Done: ", n
	for n in inodes:
		inodes[n].friends = list(map(lambda f: rev[f], inodes[n].friends))
	#   if n % 1000 == 0:
	#   	print "Done: ", n
	return inodes

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
	while bfs(graph,destination)[source] == -1:
		source = int(random.random()*graph_size)
		destination = int(random.random()*graph_size)
	print("ROUTING FROM ", source, "TO", destination)
	curr = source
	hops = 0
	while curr != destination:
		#print(hops, curr)
		neighbors = graph[curr].friends
		if destination in neighbors:
			hops+=1
			break
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
			features = (features.astype(float)-bottom)/top
			#print (weights, features)
			#print(weights,features)
			neighbor_totals[i] = theta(np.dot(weights.astype(float),features.astype(float)))
		#print (neighbor_totals)
		neighbor_totals = np.array(neighbor_totals)/sum(neighbor_totals)
		#print(neighbor_totals)
		r = random.random()
		s = 0
		i=0
		while s<r and i<len(neighbor_totals):
			s+=neighbor_totals[i]
			i+=1
		if i==0:
			i=1
		curr=neighbors[i-1]
		hops += 1
	return hops, bfs(graph,destination)[source]

def simulation_delegate(graph, weights, ret, pnum):
	os.system("taskset -p -c " + str(pnum) + " " + str(os.getpid()))
	ret[pnum] = simulation(graph,weights)


def monte_carlo(graph,weights,iters):
	stretch = np.zeros(iters)
	for i in range(iters):
		route, shortest = simulation(graph,weights)
		stretch[i] = route/shortest
	return stretch

def monte_carlo_multi(graph,weights,iters):
	stretch = np.zeros(iters)
	ps=[]
	m=Manager()
	ret = m.list([0]*len(iters))
	for i in range(iters):
		ps.append(Process(target=simulation, args=(graph, weights,ret,i)))
		print("<starting ", i, ">")
		ps[-1].start()
	for i in range(iters):
		ps[i].join()
		route, shortest = ret[i]
		stretch[i] = route/shortest
	return stretch

gowalla_weights = np.array([1.468081706129472086e-01,-1.545969938352645734e+00,-5.089143668535733855e-01,4.370629297416210868e+00,-5.072072365082430423e+00,-8.362934819850959656e+00,-4.214174338656954122e-01])
airnet_weights = np.array([-0.43524995,-2.07752073,-0.57762063,1.1298535,-1.03288428,-1.65432984,-0.14868317])

with open('data/airport_net/airnet.pkl','rb') as w:
	airnet = reindex_dict(pickle.load(w))

with open('GraphSets/test_graph.pkl','rb') as w:
	gowalla = reindex_dict(pickle.load(w))

air_aw = monte_carlo_multi(airnet, airnet_weights, 20)
air_gw = monte_carlo_multi(airnet, gowalla_weights, 20)
gow_aw = monte_carlo_multi(gowalla, airnet_weights, 20)
gow_gw = monte_carlo_multi(gowalla, gowalla_weights, 20)

np.savetxt(air_aw, "evaluations/air_aw.txt")
np.savetxt(air_gw, "evaluations/air_gw.txt")
np.savetxt(gow_aw, "evaluations/gow_aw.txt")
np.savetxt(gow_gw, "evaluations/gow_gw.txt")

