import cPickle as pickle
from copy import copy
import os
import sys
import math
import numpy as np
import random
import cProfile as p
from multiprocessing import Process
from multiprocessing import Manager
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
os.system("taskset -c 0-10 -p 0xff %d" % os.getpid())

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
	return a * (np.tanh(1./a*(dist - t))+1)

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


# def graph_to_dataset_file_xs(graph,filename,shps="shortest_paths.txt",exps="expected_paths.txt"):
# 	with open(filename,'w') as writer:
# 		for destination in graph:
# 			d=graph[destination]
# 			for neighbor in graph:
# 				n=graph[neighbor]
# 				nf = n.friends
# 				n = graph[neighbor]
# 				deg = n.deg1
# 				wdeg = math.log(deg)
# 				d = graph[destination]
# 				dist = distance(n.pos,d.pos)
# 				wdist = sig1(dist)
# 				cic = len(set(n.comm).intersection(d.comm))
# 				count = 0
# 				for _ in nf:
# 					writer.write("\t".join([str(dist),str(wdist),str(cic),str(deg),str(wdeg)]) + '\n')
# 					count+=1


def graph_to_dataset_file(graph,filename,shps="shortest_paths.txt",exps="expected_paths.txt"):
	with open(filename,'w') as writer:
		#with open(shps) as shpsr:
		with open(exps) as expsr:
			for destination in graph:
				#print 'hi', time.time()-t
				#target1_line=shpsr.readline()
				target2_line=expsr.readline()
				t1line = np.array(bfs(graph,destination))
				t2line = np.fromstring(target2_line, float, sep=' ')
				d=graph[destination]
				for source in graph:
					s=graph[source]
					sf = s.friends
					temp1=t1line[sf]
					temp2=t2line[sf]
					if len(sf)>0:
						t1 = np.argmin(temp1)
						t2 = np.argmin(temp2)
					t1n=0.
					t2n=0.
					count = 0
					for neighbor in sf:
						t1n=0.
						t2n=0.
						if temp1[count] == t1:
							t1n=1.
						if temp2[count] == t2:
							t2n=1.
						n = graph[neighbor]
						deg = n.deg1
						wdeg = math.log(deg)
						d = graph[destination]
						dist = distance(n.pos,d.pos)
						wdist = sig1(dist)
						cic = len(set(n.comm).intersection(d.comm))
						#medpower = np.median(map(lambda z: graph[z].deg1, n.friends))
						#locality = sum(map(lambda z: distance(graph[z].pos,s.pos), n.friends))
						writer.write("\t".join([str(dist),str(wdist),str(cic),str(deg),str(wdeg),str(t1n),str(t2n)]) + '\n')
						count+=1

def get_avg_and_var(dataset_file):
	avg = 0
	var = 0
	with open(dataset_file) as reader:
		line = reader.readline()
		count=1.
		while line:
			x = np.fromstring(line, dtype='float', sep=' ')
			prev_avg = np.copy(avg)
			avg += (x-avg)/count
			var += (x-prev_avg)*(x-avg)
			line = reader.readline()
			count += 1
	return avg, var/(count-1)



def normalize_dataset(dataset_file, out_file, cols):
	avg, var = get_avg_and_var(dataset_file)
	print "hi"
	with open(dataset_file) as reader:
		with open(out_file, "w") as writer:
			line = reader.readline()
			while line:
				x = np.fromstring(line, dtype='float', sep=' ')
				normal = (x[cols]-avg[cols])/(var[cols]**2)
				writer.write(" ".join(map(str, normal)))
				line=reader.readline()

def clear_comms(graph):
	for g in graph:
		graph[g].comm = []
	return graph

def switch_communities(unindexed_graph, commfile):
	with open(commfile) as reader:
		ds = []
		line = reader.readline()
		while line:
			ds.append(line.split(" "))
			line = reader.readline()

	graph = clear_comms(unindexed_graph)
	count = 0
	for d in ds:
		for u in d:
			try:
				unindexed_graph[u].comm.append(count)
			except KeyError:
				pass
		count+=1
	return unindexed_graph



# def graph_to_dataset_file1(graph,filename,shps="shortest_paths.txt",exps="expected_paths.txt"):
# 	with open(filename,'w') as writer:
# 		with open(shps) as shpsr:
# 			with open(exps) as expsr:
# 				for source in graph:
# 					s=graph[source]
# 					tar1 = map(float, shpsr.readline().split(' '))
# 					tar2 = map(float, expsr.readline().split(' '))
# 					for dest in graph:
# 						d = graph[dest]
# 						t1vals=[]
# 						t2vals=[]
# 						for neighbor in s.friends:
# 							t1vals.append(tar1[neighbor])
# 						for neighbor in s.friends:
# 							n = graph[neighbor]
# 							dist = distance(n.pos,d.pos)
# 							wdist = sig1(dist)
# 							cic = len(set(n.comm).intersection(d.comm))
# 							deg = n.deg1
# 							wdeg = math.log(deg)
# 							medpower = np.median(map(lambda z: graph[z].deg1, n.friends))
# 							locality = sum(map(lambda z: distance(graph[z].pos,s.pos), n.friends))
# 							###TODO: ADD OTHER FEATURES
# 							t1 = -1.
# 							t2 = -1.
# 							if
# 								t1 = 1.
# 								t2 = 1.
# 							writer.write("\t".join([str(dist),str(wdist),str(cic),str(deg),str(wdeg),str(medpower),str(locality),str(t1),str(t2)]) + '\n')

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


def shps_delegate(graph,writer,r):
	#print r
	#c=0
	for i in range(r[0],r[1]):
		#print " ".join(map(str, bfs(graph,i))) + "\n"
		writer.write(" ".join(map(str, bfs(graph,i))) + "\n")
		#c+=1
	writer.flush()
	#print c

def calculate_shortest_paths_to_file_multi(graph, filename, cores):
	fnames = [filename+str(q+10)+".txt" for q in range(cores)]
	writers = [open(filename+str(q+10)+".txt",'w') for q in range(cores)]
	l = len(graph)
	divs = range(0,l,int(math.ceil(float(l)/float(cores))))
	divs.append(len(graph))
	ps = []
	for w in range(len(writers)):
		ps.append(Process(target=shps_delegate, args=(graph,writers[w],(divs[w],divs[w+1]))))
		ps[-1].start()

	with open(filename+'.txt', 'w') as fw:
		for i in range(len(ps)):
			ps[i].join()
			writers[i].close()
			with open(fnames[i]) as r:
				line = r.readline()
				while line:
					#print line
					fw.write(line)
					line = r.readline()


	#os.system('touch '+filename+'.txt')
	#os.system('ls '+filename+'* | sort | while read fn ; do cat "$fn" >> '+filename+'.txt; done')
	#os.system("cat "+filename+"* > "+filename+".txt")


def calculate_expected_paths_to_files_mult(graph, filename,paths_file):
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
			if random.random()>0.5:
				friends.append(j)
		g[i]=User(None,[1,2],(random.random(),random.random()),friends)
	return g

import numpy as np

import time

t=time.time()
d = load_dict()


#d = test_graph(100)

d = switch_communities(d, "commfile.txt")



print "LOADED: ", time.time()-t
d=reindex_dict(d)
print "INDEXED: ", time.time()-t
calculate_shortest_paths_to_file_multi(d, "temp/shortest_paths",40)
print "SHPS: ", time.time()-t
calculate_expected_paths_to_file(d, "temp/expected_paths.txt", "temp/shortest_paths.txt")
print "EXPS: ", time.time()-t
p.run('graph_to_dataset_file(d,"gowalla_ml_dataset.txt")')
print "TOFILE: ", time.time()-t
normalize_dataset("gowalla_ml_dataset.txt", "gowalla_ml_dataset_norm.txt", range(5))
print "NORMALIZED: ", time.time()-t