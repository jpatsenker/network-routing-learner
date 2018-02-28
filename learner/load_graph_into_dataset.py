import cPickle as pickle
from copy import copy
import os
import sys
import math
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from core.user import User

def deg_to_rad(deg):
	return deg * math.pi / 180.0

EARTH_RAD_KM = 6371.0

def sig1(dist, a=0.001, t=275):
	return np.apply_along_axis(lambda d: a * (d - t), 0, dist)

def distance(p0, p1, rad = EARTH_RAD_KM):
	p0r = (deg_to_rad(p0[0]),deg_to_rad(p0[1]))
	p1r = (deg_to_rad(p1[0]),deg_to_rad(p1[1]))
	dlat = abs(p0r[0]-p1r[0])
	dlon = abs(p0r[1]-p1r[1])
	a = math.sin(dlat/2.)**2 + math.cos(p0r[0]) * math.cos(p1r[0]) * math.sin(dlon / 2.)**2
	d_sigma = 2. * math.asin(math.sqrt(a))
	return rad*d_sigma

def load_dict():
	nodes = pickle.load(open("GraphSets/nodes_data_dictionary.pkl", 'r'))
	return nodes

def graph_to_dataset_file(graph,filename):
	with open(filename,'w') as writer:
		for source in graph:
			s=graph[source]
			for dest in graph:
				d = graph[dest]
				for neighbor in s.friends:
					n = graph[neighbor]
					dist = d.distance(n.loc,d.loc)
					wdist = sig1(dist)
					cic = len(set(n.comm).intersection(d.comm))
					deg = n.deg1
					wdeg = math.log(deg)
					###TODO: ADD OTHER FEATURES
					writer.write('\t'.join([dist,wdist,cic,deg,wdeg]))

def bfs(graph,i):
	bfo = [-1] * len(graph)
	q = [i]
	l = i
	count = 0
	while len(q)>0:
		z = q.pop()
		bfo[z] = count
		q.extend(graph[z].friends)
		if z==l:
			count+=1
			l=q[-1]
	return bfo

def calculate_shortest_paths_to_file(graph, filename):
	with open(filename, 'w') as writer:
		for i in graph:
			writer.write("\t".join(bfs(graph,i)) + "\n")

def calculate_expected_paths_to_file(graph, filename):
	with open(filename, 'w') as writer:
		for i in graph:
			paths = bfs(graph,i)
			c=0
			while c<10:
				pathsnew = [0]*len(paths)
				for s in graph:
					sgf = graph[s].friends
					if d in sgf:
						pathsnew[s] = 1.
						continue
					pathsnew[s] = sum(map(lambda x: 1./len(sgf)*paths[x] + 1, sgf))
					paths=copy(pathsnew)
				c+=1
			writer.write("\t".join(paths) + "\n")




d = load_dict()
graph_to_dataset_file(d,"gowalla_ml_dataset.txt")