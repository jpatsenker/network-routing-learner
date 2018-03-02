import cPickle as pickle
from copy import copy
import os
import sys
import math
import numpy as np
import random
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from core.user import User

def reindex_dict(nodes):
	inodes = dict()
	ndic = dict(zip(range(len(nodes.keys())), nodes.keys()))
	rev = {v: k for k, v in ndic.iteritems()}
	for n in range(len(nodes)):
		inodes[n] = copy(nodes[ndic[n]])
		inodes[n].uid = n
	#   if n % 1000 == 0:
	#       print "Done: ", n
	for n in inodes:
		inodes[n].friends = map(lambda f: rev[f], inodes[n].friends)
	#   if n % 1000 == 0:
	#   	print "Done: ", n
	return inodes

def adj_list(graph):
	al = []
	for i in graph:
		al.append(np.array([0]*len(graph)))
		for f in graph[i].friends:
			al[-1][f]=1
	return np.array(al)

def deg_to_rad(deg):
	return deg * math.pi / 180.0

EARTH_RAD_KM = 6371.0

def sig1(dist, a=500., t=1000.):
	return a * np.tanh(1./a*(dist - t))

def distance(p0, p1, rad = EARTH_RAD_KM):
	p0r = (deg_to_rad(p0[0]),deg_to_rad(p0[1]))
	p1r = (deg_to_rad(p1[0]),deg_to_rad(p1[1]))
	dlat = abs(p0r[0]-p1r[0])
	dlon = abs(p0r[1]-p1r[1])
	a = math.sin(dlat/2.)**2 + math.cos(p0r[0]) * math.cos(p1r[0]) * math.sin(dlon / 2.)**2
	d_sigma = 2. * math.asin(math.sqrt(a))
	return rad*d_sigma

def load_dict():
	nodes = pickle.load(open("GraphSets/test_graph.pkl", 'r'))
	return nodes

def graph_to_dataset_file(graph,filename,shps="shortest_paths.txt",exps="expected_paths.txt"):
	with open(filename,'w') as writer:
		with open(shps) as shpsr:
			with open(exps) as expsr:
				for source in graph:
					s=graph[source]
					tar1 = map(float, shpsr.readline().split(' '))
					tar2 = map(float, expsr.readline().split(' '))
					for dest in graph:
						d = graph[dest]
						for neighbor in s.friends:
							n = graph[neighbor]
							dist = distance(n.pos,d.pos)
							wdist = sig1(dist)
							cic = len(set(n.comm).intersection(d.comm))
							deg = n.deg1
							wdeg = math.log(deg)
							medpower = np.median(map(lambda z: graph[z].deg1, n.friends))
							locality = sum(map(lambda z: distance(graph[z].pos,graph[s].pos), n.friends))
							###TODO: ADD OTHER FEATURES
							t1 = tar1[dest]
							t2 = tar2[dest]
							writer.write("\t".join([str(dist),str(wdist),str(cic),str(deg),str(wdeg),str(medpower),str(locality),str(t1),str(t2)]) + '\n')

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

def read_bfs(filename,i):
	with open(filename) as r:
		line=r.readline()
		c=0
		while c<i:
			line=r.readline()
			c+=1
		return map(int,line.split(' '))

def calculate_shortest_paths_to_file(graph, filename):
	with open(filename, 'w') as writer:
		for i in graph:
			writer.write(" ".join(map(str, bfs(graph,i))) + "\n")

def calculate_expected_paths_to_files_mult(graph, filename,paths_file):
	#with open(filename, 'w') as writer:
	writer = [open('test/'+filename+str(q),'w') for q in range(20)]
	for i in graph:
		paths = []
		paths = read_bfs(paths_file,i)
		c = 0
		while c < 10:
			pathsnew = [0]*len(paths)
			for s in graph:
				sgf = graph[s].friends
				if s == i:
					pathsnew[s] = 0.
					continue
				if i in sgf:
					pathsnew[s] = 1.
					continue
				pathsnew[s] = sum(map(lambda x:paths[x], sgf))*1./len(sgf)+1.
				paths=copy(pathsnew)
			writer[c].write("\t".join(map(str, paths)) + "\n")
			c+=1

def calculate_expected_paths_to_file(graph,filename,paths_file):
	with open(filename,'w') as writer:
		for d in graph:
			paths = read_bfs(paths_file,d)
			iter=0
			while iter<10:
				new_paths=[0]*len(paths)
				for s in graph:
					sfriends = graph[s].friends
					if s==d:
						new_paths[s]=0.
						continue
					if d in sfriends:
						new_paths[s]=1.
						continue
					new_paths[s] = sum(map(lambda f: (paths[f]+1)*1./len(sfriends),sfriends))
				paths=copy(new_paths)
				iter+=1
			writer.write(' '.join(map(str,paths)) + '\n')


def test_graph(n):
	g={}
	for i in range(n):
		friends = []
		for j in range(i):
			if i in g[j].friends:
				friends.append(j)
		for j in range(i+1,n):
			if random.random()>.95:
				friends.append(j)
		g[i]=User(None,(random.random(),random.random()),[1,2],friends)
	return g

import numpy as np

import time

t=time.time()
#d = load_dict()

d=test_graph(100)


print "LOADED: ", time.time()-t
d=reindex_dict(d)
print "INDEXED: ", time.time()-t
calculate_shortest_paths_to_file(d, "shortest_paths.txt")
print "SHPS: ", time.time()-t
calculate_expected_paths_to_file(d, "expected_paths.txt", "shortest_paths.txt")
print "EXPS: ", time.time()-t
graph_to_dataset_file(d,"gowalla_ml_dataset.txt")
print "TOFILE: ", time.time()-t

