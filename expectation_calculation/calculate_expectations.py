import sys
sys.path.append("../..")
from core.connection import Edge
from core.user import User
#from gowalla_research.core.connection import Edge
#from gowalla_research.core.user import User
from createProbabilityMap import createPMapCommunities4Dest
from createProbabilityMap import createPMapRandom4Dest
from createProbabilityMap import createPMapDistance4Dest
from createProbabilityMap import createPMapDegree4Dest

import cPickle as pickle
import copy
import time
#from edge import StatEdge


def calculate_expectations_very_pickled(us,es,method,proot,pOut):
	#print "START", 0
	stime = time.time()
	usersDict = pickle.load(open(us,"r"))
	edges = pickle.load(open(es,"r"))
	users = usersDict.values()
	backToUUIDS = usersDict.keys()
	ktoi = dict(zip(usersDict.keys(),range(len(usersDict.keys()))))
	#print "SETUP DONE: ", time.time()-stime
	pmap_random = None
	pmap_degree = None
	if method == "random":
		pmap_random = pickle.load(open(proot+"_equal_weighted.pkl","r"))
	elif method == "degree":
		pmap_degree = pickle.load(open(proot+"_degree_weighted.pkl","r"))
	bigSumA = 0
	bigSumB = 0
	eee = 0
	#print "Done Setup"
	for d in range(len(users)):
		probabilityMap = {}
		if method == "random":
			probabilityMap = pmap_random
		elif method == "distance":
			probabilityMap = pickle.load(open(proot + "_distance_weighted_" + str(backToUUIDS[d]) + ".pkl", "r"))
		elif method == "communities":
			probabilityMap = pickle.load(open(proot + "_community_weighted_" + str(backToUUIDS[d]) + ".pkl", "r"))
		elif method == "degree":
			probabilityMap = pmap_degree
		else:
			print "INVALID METHOD"
			return

		hopsToDestination = [-1 for i in range(len(users))]
		bfsqueue = [d]
		curr_layer = 0
		last_in_layer = d
		visited = [False for i in range(len(users))]
		visited[d] = True
		#print "SETUP2 DONE: ", time.time()-stime
		while len(bfsqueue) > 0:
			hopsToDestination[bfsqueue[0]] = curr_layer
			l = None
			for f in users[bfsqueue[0]].friends:
				fi = ktoi[f]
				if not visited[fi]:
					bfsqueue.append(fi)
					visited[fi] = True
					l = fi
				if bfsqueue[0] == last_in_layer:
					last_in_layer = l
					curr_layer += 1
			bfsqueue = bfsqueue[1:]

		#print "OPT CALC DONE: ", time.time()-stime
		optimal = copy.deepcopy(hopsToDestination)
		#print "OPT COPY DONE: ", time.time()-stime
		#do 10 times till converge
		for i in range(10):
			nextHD = [0 for i in range(len(hopsToDestination))]
			for s in range(len(hopsToDestination)):
				if backToUUIDS[d] in users[s].friends:
					nextHD[s] = 1
					continue
				nextHD[s] = sum(map(lambda i: probabilityMap[backToUUIDS[s]][i]*hopsToDestination[ktoi[i]],users[s].friends))+1
			hopsToDestination = nextHD
		#print "ECALC DONE: ", time.time()-stime
		#print hopsToDestination


		for s in range(len(hopsToDestination)):
			if s == d:
				continue
			bigSumA += hopsToDestination[s] - optimal[s]
			bigSumB += hopsToDestination[s]/optimal[s]
			eee+=1
		#print hopsToDestination == optimal
		#print float(bigSumA)/float(eee)
		#print float(bigSumB)/float(eee)
		#print d
		#print "NUM CALC DONE: ", time.time()-stime

	outs = {"percentage": float(bigSumB) / float(eee), "subtraction": float(bigSumA) / float(eee)}

	pickle.dump(outs, open(pOut, "w"))
	print "DONE: ", time.time()-stime

def calculate_expectations_very_pickled_1Dest(us,es,method,proot,pOut,seed):
	#print "START", 0
	stime = time.time()
	usersDict = pickle.load(open(us,"r"))
	edges = pickle.load(open(es,"r"))
	users = usersDict.values()
	backToUUIDS = usersDict.keys()
	ktoi = dict(zip(usersDict.keys(),range(len(usersDict.keys()))))
	#print "SETUP DONE: ", time.time()-stime
	pmap_random = None
	pmap_degree = None
	if method == "random":
		pmap_random = pickle.load(open(proot+"_equal_weighted.pkl","r"))
	elif method == "degree":
		pmap_degree = pickle.load(open(proot+"_degree_weighted.pkl","r"))
	bigSumA = 0
	bigSumB = 0
	eee = 0
	#print "Done Setup"
	for d in range(seed,seed+1):
		probabilityMap = {}
		if method == "random":
			probabilityMap = pmap_random
		elif method == "distance":
			probabilityMap = pickle.load(open(proot + "_distance_weighted_" + str(backToUUIDS[d]) + ".pkl", "r"))
		elif method == "communities":
			probabilityMap = pickle.load(open(proot + "_community_weighted_" + str(backToUUIDS[d]) + ".pkl", "r"))
		elif method == "degree":
			probabilityMap = pmap_degree
		else:
			print "INVALID METHOD"
			return

		hopsToDestination = [-1 for i in range(len(users))]
		bfsqueue = [d]
		curr_layer = 0
		last_in_layer = d
		visited = [False for i in range(len(users))]
		visited[d] = True
		#print "SETUP2 DONE: ", time.time()-stime
		while len(bfsqueue) > 0:
			hopsToDestination[bfsqueue[0]] = curr_layer
			l = None
			for f in users[bfsqueue[0]].friends:
				fi = ktoi[f]
				if not visited[fi]:
					bfsqueue.append(fi)
					visited[fi] = True
					l = fi
				if bfsqueue[0] == last_in_layer:
					last_in_layer = l
					curr_layer += 1
			bfsqueue = bfsqueue[1:]
		#print "OPT CALC DONE: ", time.time()-stime
		optimal = copy.deepcopy(hopsToDestination)
		#print "OPT COPY DONE: ", time.time()-stime
		#do 10 times till converge
		for i in range(10):
			nextHD = [0 for i in range(len(hopsToDestination))]
			for s in range(len(hopsToDestination)):
				if backToUUIDS[d] in users[s].friends:
					nextHD[s] = 1
					continue
				nextHD[s] = sum(map(lambda i: probabilityMap[backToUUIDS[s]][i]*hopsToDestination[ktoi[i]],users[s].friends))+1
			hopsToDestination = nextHD
		#print "ECALC DONE: ", time.time()-stime
		#print hopsToDestination


		for s in range(len(hopsToDestination)):
			if s == d:
				continue
			bigSumA += hopsToDestination[s] - optimal[s]
			bigSumB += hopsToDestination[s]/optimal[s]
			eee+=1
		#print hopsToDestination == optimal
		#print float(bigSumA)/float(eee)
		#print float(bigSumB)/float(eee)

	outs = {"percentage": float(bigSumB) / float(eee), "subtraction": float(bigSumA) / float(eee)}

	pickle.dump(outs, open(pOut, "w"))
	return outs
	#print "DONE: ", time.time()-stime


def calculate_expectations(us,es,method,pOut):
	#print "START", 0
	stime = time.time()
	usersDict = pickle.load(open(us,"r"))
	edges = pickle.load(open(es,"r"))
	users = usersDict.values()
	backToUUIDS = usersDict.keys()
	ktoi = dict(zip(usersDict.keys(),range(len(usersDict.keys()))))
	#print "SETUP DONE: ", time.time()-stime
	pmap_random = createPMapRandom4Dest(usersDict,edges,backToUUIDS[0])
	pmap_degree = createPMapDegree4Dest(usersDict,edges,backToUUIDS[0])
	bigSumA = 0
	bigSumB = 0
	eee = 0
	print "Done Setup"
	for d in range(len(users)):
		probabilityMap = {}
		if method == "random":
			probabilityMap = pmap_random
		elif method == "distance":
			probabilityMap = createPMapDistance4Dest(usersDict,edges,backToUUIDS[d])
		elif method == "communities":
			probabilityMap = createPMapCommunities4Dest(usersDict,edges,backToUUIDS[d])
		elif method == "degree":
			probabilityMap = pmap_degree
		else:
			print "INVALID METHOD"
			return

		hopsToDestination = [-1 for i in range(len(users))]
		bfsqueue = [d]
		curr_layer = 0
		last_in_layer = d
		visited = [False for i in range(len(users))]
		visited[d] = True
		#print "SETUP2 DONE: ", time.time()-stime
		while len(bfsqueue) > 0:
			hopsToDestination[bfsqueue[0]] = curr_layer
			l = None
			for f in users[bfsqueue[0]].friends:
				fi = ktoi[f]
				if not visited[fi]:
					bfsqueue.append(fi)
					visited[fi] = True
					l = fi
				if bfsqueue[0] == last_in_layer:
					last_in_layer = l
					curr_layer += 1
			bfsqueue = bfsqueue[1:]
		#print "OPT CALC DONE: ", time.time()-stime
		optimal = copy.deepcopy(hopsToDestination)
		#print "OPT COPY DONE: ", time.time()-stime
		#do 10 times till converge
		for i in range(10):
			nextHD = [0 for i in range(len(hopsToDestination))]
			for s in range(len(hopsToDestination)):
				if backToUUIDS[d] in users[s].friends:
					nextHD[s] = 1
					continue
				nextHD[s] = sum(map(lambda i: probabilityMap[backToUUIDS[s]][i]*hopsToDestination[ktoi[i]],users[s].friends))+1
			hopsToDestination = nextHD
		#print "ECALC DONE: ", time.time()-stime
		#print hopsToDestination


		for s in range(len(hopsToDestination)):
			if s == d:
				continue
			bigSumA += hopsToDestination[s] - optimal[s]
			bigSumB += hopsToDestination[s]/optimal[s]
			eee+=1
		#print hopsToDestination == optimal
		#print float(bigSumA)/float(eee)
		#print float(bigSumB)/float(eee)

	outs = {"percentage": float(bigSumB) / float(eee), "subtraction": float(bigSumA) / float(eee)}

	pickle.dump(outs, open(pOut, "w"))
	#print "DONE: ", time.time()-stime






def calculate_expectations_for_user_pickled(us,es,weight_data,pOut):
	print "START", 0
	stime = time.time()
	usersDict = pickle.load(open(us,"r"))
	edges = pickle.load(open(es,"r"))
	users = usersDict.values()
	backToUUIDS = usersDict.keys()
	ktoi = dict(zip(usersDict.keys(),range(len(usersDict.keys()))))
	print "SETUP DONE: ", time.time()-stime
	d = 0
	hopsToDestination = [-1 for i in range(len(users))]
	bfsqueue = []
	bfsqueue.append(d)
	curr_layer = 0
	last_in_layer = d
	visited = [False for i in range(len(users))]
	visited[d] = True
	print "SETUP2 DONE: ", time.time()-stime
	while len(bfsqueue)>0:
	#print BFSQueue
		hopsToDestination[bfsqueue[0]]=curr_layer
		l = None
		for f in users[bfsqueue[0]].friends:
			fi = ktoi[f]
			if not visited[fi]:
				bfsqueue.append(fi)
				visited[fi] = True
				l=fi
				if bfsqueue[0] == last_in_layer:
					last_in_layer = l
					curr_layer += 1
			bfsqueue = bfsqueue[1:]
	print "OPT CALC DONE: ", time.time()-stime
	optimal = copy.deepcopy(hopsToDestination)
	print "OPT COPY DONE: ", time.time()-stime
	probabilityMap = pickle.load(open(weight_data,"r"))	
	print "LOAD PMAP DONE: ", time.time()-stime
	#do 10 times till converge

	for i in range(10):
		nexthd = [0 for i in range(len(hopsToDestination))]
		for s in range(len(hopsToDestination)):
			if backToUUIDS[d] in users[s].friends:
				nexthd[s] = 1
				continue
			nexthd[s] = sum(map(lambda i: probabilityMap[backToUUIDS[d]][backToUUIDS[s]][i]*hopsToDestination[ktoi[i]],users[s].friends))+1
		hopsToDestination = nexthd
	print "ECALC DONE: ", time.time()-stime
	#print hopsToDestination

	bigSumA = 0
	bigSumB = 0
	eee = 0
	for s in range(len(hopsToDestination)):
		if s is d:
			continue
		bigSumA += hopsToDestination[s] - optimal[s]
		bigSumB += hopsToDestination[s]/optimal[s]
		eee+=1
	#print hopsToDestination == optimal
	print float(bigSumA)/float(eee)
	print float(bigSumB)/float(eee)
	
	outs = {}
	outs["percentage"] = float(bigSumB)/float(eee)
	outs["subtraction"] = float(bigSumA)/float(eee)
	
	pickle.dump(outs, open(pOut, "w"))
	print "DONE: ", time.time()-stime


def calculate_expectations_pickled(us,es,weight_data,pOut):
	print "START", 0
	stime = time.time()
	usersDict = pickle.load(open(us,"r"))
	edges = pickle.load(open(es,"r"))
	users = usersDict.values()
	backToUUIDS = usersDict.keys()
	ktoi = dict(zip(usersDict.keys(),range(len(usersDict.keys()))))
	print "SETUP DONE: ", time.time()-stime
	probabilityMap = pickle.load(open(weight_data,"r"))
	print "LOAD PMAP DONE: ", time.time()-stime
	for d in range(len(users)):
		hopsToDestination = [-1 for i in range(len(users))]
		bfsqueue = [d]
		curr_layer = 0
		last_in_layer = d
		visited = [False for i in range(len(users))]
		visited[d] = True
		print "SETUP2 DONE: ", time.time()-stime
		while len(bfsqueue)>0:
			#print BFSQueue
			hopsToDestination[bfsqueue[0]]=curr_layer
			l = None
			for f in users[bfsqueue[0]].friends:
				fi = ktoi[f]
				if not visited[fi]:
					bfsqueue.append(fi)
					visited[fi] = True
					l=fi
				if bfsqueue[0] == last_in_layer:
					last_in_layer = l
					curr_layer += 1
			bfsqueue = bfsqueue[1:]
		print "OPT CALC DONE: ", time.time()-stime
		optimal = copy.deepcopy(hopsToDestination)
		print "OPT COPY DONE: ", time.time()-stime
		#do 10 times till converge

		for i in range(10):
			nextHD = [0 for i in range(len(hopsToDestination))]
			for s in range(len(hopsToDestination)):
				if backToUUIDS[d] in users[s].friends:
					nextHD[s] = 1
					continue
				nextHD[s] = sum(map(lambda i: probabilityMap[backToUUIDS[d]][backToUUIDS[s]][i]*hopsToDestination[ktoi[i]],users[s].friends))+1
			hopsToDestination = nextHD
		print "ECALC DONE: ", time.time()-stime
		#print hopsToDestination

		bigSumA = 0
		bigSumB = 0
		eee = 0
		for s in range(len(hopsToDestination)):
			if s is d:
				continue
			bigSumA += hopsToDestination[s] - optimal[s]
			bigSumB += hopsToDestination[s]/optimal[s]
			eee+=1
		#print hopsToDestination == optimal
		print float(bigSumA)/float(eee)
		print float(bigSumB)/float(eee)

		outs = {}
		outs["percentage"] = float(bigSumB)/float(eee)
		outs["subtraction"] = float(bigSumA)/float(eee)

		pickle.dump(outs, open(pOut, "w"))
		print "DONE: ", time.time()-stime





def calculate_expectations_depr(us,es,weight_data,pOut):
	users = pickle.load(open(us,"r"))
	edges = pickle.load(open(es,"r"))

	hopsToDestination = {}

	for d in users:
		#print d
		hopsToDestination[d] = {}
		BFSQueue = []
		BFSQueue.append(d)
		curr_layer = 0
		last_in_layer = d
		visited = [d]
		while len(BFSQueue)>0:
				#print BFSQueue
			hopsToDestination[d][BFSQueue[0]]=curr_layer
			l = None
			for f in users[BFSQueue[0]].friends:
				if f not in visited:
					BFSQueue.append(f)
					visited.append(f)
					l=f
				if BFSQueue[0] == last_in_layer:
					last_in_layer = l
					curr_layer += 1	
			BFSQueue = BFSQueue[1:]
	
	optimal = copy.deepcopy(hopsToDestination)
	print "FINISHED OPTIMAL"
	probabilityMap = pickle.load(open(weight_data,"r"))
	#probabilityMap = pickle.load(open("pmap_random.pkl","r"))
	#probabilityMap = pickle.load(open("pmap_community_weighted.pkl","r"))
	
	#for d in edges:
	#	probabilityMap[d] = {}
	#	for s in edges[d]:
	#		probabilityMap[d][edges[d][s].uid2] = 1.0/float(users[d].deg1)
	
	#print hopsToDestination
	#print probabilityMap
	
	#do 10 times till converge
	for i in range(10):
		nextHD = {}
		for d in hopsToDestination:
			nextHD[d] = {}
			for s in hopsToDestination[d]:
				if d in users[s].friends:
					nextHD[d][s] = 1
					continue
				nextHD[d][s] = sum(map(lambda i: probabilityMap[d][s][i]*hopsToDestination[d][i],users[s].friends))+1
		hopsToDestination = nextHD
	
	#print hopsToDestination

	bigSumA = 0
	bigSumB = 0
	eee = 0
	for d in hopsToDestination:
		for s in hopsToDestination[d]:
			if s is d:
				continue
			bigSumA += hopsToDestination[d][s] - optimal[d][s]
			bigSumB += hopsToDestination[d][s]/optimal[d][s]
			eee+=1
	#print hopsToDestination == optimal
	print float(bigSumA)/float(eee)
	print float(bigSumB)/float(eee)

	outs = {}
	outs["percentage"] = float(bigSumB)/float(eee)
	outs["subtraction"] = float(bigSumA)/float(eee)

	pickle.dump(outs, open(pOut, "w"))

