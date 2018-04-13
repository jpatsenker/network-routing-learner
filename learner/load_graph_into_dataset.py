import pickle as pickle
from copy import copy
import os
import sys
import math
import numpy as np
import random
import cProfile as p
import pandas as pd
from multiprocessing import Process
from multiprocessing import Manager
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))


from core.user import User

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
	nodes = pickle.load(open('GraphSets/test_graph.pkl','rb'))
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


def graph_to_dataset_file(graph,filename,shps,exps):
	with open(filename,'w') as writer:
		#with open(shps) as shpsr:
		for destination in graph:
			print(destination,time.time())
			#target1_line=shpsr.readline()
			t1line = shps[destination]
			t2line = exps[destination]
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
					#print(neighbor,n.friends)
					medpower = np.median(list(map(lambda z: graph[z].deg1, n.friends)))
					locality = sum(map(lambda z: distance(graph[z].pos,s.pos), n.friends))
					writer.write("\t".join([str(dist),str(wdist),str(cic),str(deg),str(wdeg),str(medpower),str(locality),str(t1n),str(t2n)]) + '\n')
					count+=1

def dataset_delegate(graph,shps,exps,r,writer):
	sc=0
	for destination in range(r[0],r[1]):
		#print(destination)
		t1line = shps[destination-r[0]]
		t2line = exps[destination-r[0]]
		d=graph[destination]
		for source in graph:
			s=graph[source]
			sf = s.friends
			temp1=t1line[sf]
			temp2=t2line[sf]
			if len(sf)>0:
				t1 = np.argmin(temp1)
				t2 = np.argmin(temp2)
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
				medpower = np.median(list(map(lambda z: graph[z].deg1, n.friends)))
				locality = sum(map(lambda z: distance(graph[z].pos,s.pos), n.friends))
				writer.write("\t".join([str(dist),str(wdist),str(cic),str(deg),str(wdeg),str(medpower),str(locality),str(t1n),str(t2n)]) + '\n')
				count+=1
				sc+=1
	writer.close()

def graph_to_dataset_file_multi(graph,filename,shps,exps,cores):
	writers = [open(filename+str(q)+".txt",'w') for q in range(cores)]
	ps = []
	l=len(graph)
	divs = list(range(0,l,int(math.ceil(float(l)/float(cores)))))
	divs.append(len(graph))
	for w in range(len(writers)):
		ps.append(Process(target=dataset_delegate, args=(graph,shps,exps,(divs[w],divs[w+1]),writers[w])))
		ps[-1].start()
	for p in range(len(ps)):
		ps[p].join()
	finwrite=open(filename + '.txt','w')
	readers = [open(filename+str(q)+".txt",'r') for q in range(len(writers))]
	for r in readers:
		s=r.read()
		finwrite.write(s)
		r.close()
	finwrite.close()




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
	with open(dataset_file) as reader:
		with open(out_file, "w") as writer:
			line = reader.readline()
			while line:
				x = np.fromstring(line, dtype='float', sep=' ')
				normal = (x[cols]-avg[cols])/(var[cols]**2)
				writer.write(" ".join(list(map(str, normal))))
				line=reader.readline()

def normalize_dataset_multi(dataset_file, out_file, cols, cores):
	avg, var = get_avg_and_var(dataset_file, cores)
	with open(dataset_file) as reader:
		with open(out_file, "w") as writer:
			line = reader.readline()
			while line:
				x = np.fromstring(line, dtype='float', sep=' ')
				normal = (x[cols]-avg[cols])/(var[cols]**2)
				writer.write(" ".join(list(map(str, normal))))
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


def shps_delegate(graph,writer,r, q):
	os.system("taskset -p -c " + str(q) + " " + str(os.getpid()))
	#print r
	#c=0
	for i in range(r[0],r[1]):
		#print " ".join(map(str, bfs(graph,i))) + "\n"
		writer.write(" ".join(map(str, bfs(graph,i))) + "\n")
		#c+=1
		# if i%10==0:
		# 	print "PROCESS", os.getpid(), "HAS COMPLETED", i, "BFSs"
		# 	sys.stdout.flush()
	writer.flush()
	#print c

def calculate_shortest_paths_to_file_multi(graph, filename, cores):
	fnames = [filename+str(q+10)+".txt" for q in range(cores)]
	writers = [open(filename+str(q+10)+".txt",'w') for q in range(cores)]
	l = len(graph)
	divs = list(range(0,l,int(math.ceil(float(l)/float(cores)))))
	divs.append(len(graph))
	ps = []
	for w in range(len(writers)):
		ps.append(Process(target=shps_delegate, args=(graph,writers[w],(divs[w],divs[w+1]), w)))
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


# def nav_readers_to_divs(reader,divs):
# 	c=1
# 	lines = map(lambda r: r.readline(), reader)
# 	while l:
#
# 		lines = map(lambda r: r.readline(), reader)
# 	map(lambda r: r.close(), reader)

def exps_delegate(graph,prev_paths,next_paths,r,q):
	#os.system("taskset -p -c " + str(q) + " " + str(os.getpid()))
	for i in range(r[0],r[1]):
		for s in graph:
			sfriends = list(graph[s].friends)
			if s==i:
				next_paths[i,s] = 0.
				continue
			if i in sfriends:
				next_paths[i,s] = 1.
				continue
			l = len(sfriends)
			next_paths[i,s] = sum(map(lambda f: (prev_paths[i,f]+1)*1./l,sfriends))


def calculate_expected_paths_mult(graph, filename,prev_paths,cores):
	# fnames = [filename+str(q+10)+".txt" for q in range(cores)]
	# writer = [open(filename+str(q+10)+".txt",'w') for q in range(cores)]
	# reader = [open(filename+str(q+10)+".txt",'r') for q in range(cores)]
	l = len(graph)
	divs = list(range(0,l,int(math.ceil(float(l)/float(cores)))))
	divs.append(l)
	#nav_readers_to_divs(reader, divs)
	c=0
	while c < 10:
		ps = []
		q=0
		for d in range(len(divs)-1):
			next_paths = np.copy(prev_paths)
			ps.append(Process(target = exps_delegate, args = (graph,prev_paths,next_paths,(divs[d],divs[d+1]),q)))
			ps[-1].start()
			q+=1
		for p in range(len(ps)):
			ps[p].join()
		c+=1
		print("Iter", c, "complete")
	return next_paths

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
			if random.random()<1./float(n):
				friends.append(j)
		g[i]=User(None,[1,2],(random.random(),random.random()),friends)
	return g





import time

t=time.time()
d = load_dict()


#d = test_graph(1000)

d = switch_communities(d, "commfile.txt")



print ("LOADED: ", time.time()-t)
d=reindex_dict(d)
print ("INDEXED: ", time.time()-t)
#calculate_shortest_paths_to_file_multi(d, "temp/shortest_paths",50)
#shps = pd.io.parsers.read_csv("temp/shortest_paths.txt",sep=' ',header=None,engine='c',dtype='float32').as_matrix()
shps_str = open("temp/shortest_paths.txt").read()
print ("READ SHPS: ", time.time()-t)
shps_arr = shps_str.split("\n")
print ("SPLIT ROWS: ", time.time()-t)
shps_darr = map(lambda x: x.split(' '), shps_arr)
shps_darr=list(shps_darr)
print ("SPLIT COLS: ", time.time()-t)
shps = np.array(shps_darr[:-1])
print ("AS NUMPY: ", time.time()-t)
del shps_str, shps_arr, shps_darr
print ("DELETE: ", time.time()-t)
shps = shps.astype('int')
print ("SHPS: ", time.time()-t)
exps=calculate_expected_paths_mult(d, "temp/expected_paths.txt", shps, 50)
print ("EXPS: ", time.time()-t)
graph_to_dataset_file_multi(d,"temp/gowalla_ml_dataset",shps,exps,50)
print ("TOFILE: ", time.time()-t)
normalize_dataset("temp/gowalla_ml_dataset.txt", "temp/gowalla_ml_dataset_norm.txt", range(5))
print ("NORMALIZED: ", time.time()-t)