import sys
sys.path.append("../..")

import cPickle as pickle
from core.user import User
from core.connection import Edge
from core.connection import distance
#from gowalla_research.core.user import User
#from gowalla_research.core.connection import Edge
#from gowalla_research.core.connection import distance
import time


def createPMapDistanceDatabase(users,edges,connection,gname):
	database = connection.cursor()
	tname = "distance" + gname
	database.execute("CREATE TABLE " + tname + " (destination VARCHAR(20), source VARCHAR(20), next VARCHAR(20), probability FLOAT);")
	database.execute("CREATE INDEX destination_i ON " + tname + " (destination)")
	database.execute("CREATE INDEX source_i ON " + tname + " (source)")
	database.execute("CREATE INDEX next_i ON " + tname + " (next)")
	for d in users:
		for s in edges:
			denom = sum(map(lambda x: 1.0/float(distance(users[x.uid2].pos, users[d].pos) + 1.0), edges[s].values()))
			for f in edges[s]:
				p = (1.0 / float(distance(users[f].pos, users[d].pos) + 1.0))/denom
				database.execute("INSERT INTO " + tname + " (destination,source,next,probability) VALUES (" + d + "," + s + "," + f + "," + str(p) + ")", params=None, multi=False)
				connection.commit()
				#print "INSERT INTO " + tname + " (destination,source,next,probability) VALUES (" + d + "," + s + "," + f + "," + str(p) + ")"
		print "Finished Dest " + str(d) + " " + str(time.time())


# def createPMapDistanceFast(users, edges):
# 	allpmaps = []
# 		usersList = users.values()
# 		userKey = users.keys()
# 		probabilityMap = []
# 		for s in edges:
# 			probabilityMap.append([])
# 			denom = sum(map(lambda x: 1.0 / float(distance(users[x.uid2].pos, usersList.pos) + 1.0), edges[d].values()))

def createPMapDistance4Dest(users,edges,dest):
	probabilityMap = {}
	for d in edges:
		probabilityMap[d] = {}
		denom = sum(map(lambda x: 1.0 / float(distance(users[x.uid2].pos, users[dest].pos) + 1.0), edges[d].values()))
		for s in edges[d]:
			probabilityMap[d][s] = (1.0 / float(distance(users[s].pos, users[dest].pos) + 1.0))/denom
	return probabilityMap

def createPMapCommunities4Dest(users,edges,dest):
	probabilityMap = {}
	for d in edges:
		probabilityMap[d] = {}
		denom = sum(map(lambda x: 1.0 / float(distance(users[x.uid2].pos, users[dest].pos) + 1.0), edges[d].values()))
		for s in edges[d]:
			probabilityMap[d][s] = (1.0 / float(distance(users[s].pos, users[dest].pos) + 1.0))/denom
	return probabilityMap

def createPMapDegree4Dest(users,edges,dest):
	probabilityMap = {}
	for d in edges:
		probabilityMap[d] = {}
		denom = sum(map(lambda x: users[x.uid2].deg1, edges[d].values()))
		for s in edges[d]:
			probabilityMap[d][s] = float(users[s].deg1)/float(denom)
	return probabilityMap

def createPMapRandom4Dest(users,edges,dest):
	probabilityMap = {}
	for d in edges:
		probabilityMap[d] = {}
		for s in edges[d]:
			probabilityMap[d][edges[d][s].uid2] = 1.0 / float(users[d].deg1)
	return probabilityMap

def createPMapDistance(users,edges):
	probabilityMap = {}
	for dest in users:
		probabilityMap[dest] = {}
		for d in edges:
			probabilityMap[dest][d] = {}
			denom = sum(map(lambda x: 1.0/float(distance(users[x.uid2].pos, users[dest].pos) + 1.0), edges[d].values()))
			for s in edges[d]:
			#print dest, d, s
				probabilityMap[dest][d][s] = (1.0/float(distance(users[s].pos, users[dest].pos) + 1.0))/denom
	return probabilityMap


def createPMapRandom(users,edges):
	probabilityMap = {}
	for dest in users:
		probabilityMap[dest] = {}
		for d in edges:
			probabilityMap[dest][d] = {}
			for s in edges[d]:
				probabilityMap[dest][d][edges[d][s].uid2] = 1.0/float(users[d].deg1)
	return probabilityMap


def createPMapCommunities(users,edges):
	probabilityMap = {}
	for dest in users:
		probabilityMap[dest] = {}
		for d in edges:
			probabilityMap[dest][d] = {}
			denom = sum(map(lambda x: len(set(users[dest].comm).intersection(set(users[x.uid2].comm)))+1.0, edges[d].values()))
			for s in edges[d]:
				#print dest, d, s
				probabilityMap[dest][d][s] = (len(set(users[dest].comm).intersection(set(users[s].comm)))+1.0)/denom
	return probabilityMap

def createPMapDegree(users,edges):
	probabilityMap = {}
	for dest in users:
		probabilityMap[dest] = {}
		for d in edges:
			probabilityMap[dest][d] = {}
			denom = sum(map(lambda x: users[x.uid2].deg1, edges[d].values()))
			for s in edges[d]:
				probabilityMap[dest][d][s] = float(users[s].deg1)/float(denom)
	return probabilityMap

def createPMapsP2D(uName,eName):
	users = pickle.load(open(uName,"r"))
	edges = pickle.load(open(eName,"r"))
	random = createPMapRandom(users, edges)
	print "Created Random for " + uName
	p_map_distance = createPMapDistance(users, edges)
	print "Created Distance for " + uName
	communities = createPMapCommunities(users, edges)
	print "Created Communities for " + uName
	degree = createPMapDegree(users, edges)
	print "Created Degree for " + uName
	return {"random": random, "distance": p_map_distance, "communities": communities, "degree": degree}

def createPMapsP2PE(uName,eName,outRoot):
	users = pickle.load(open(uName,"r"))
	edges = pickle.load(open(eName,"r"))
	degree = createPMapDegree4Dest(users, edges, None)
	random = createPMapRandom4Dest(users, edges, None)
	pickle.dump(random,open(outRoot + "_equal_weighted.pkl","w"))
	pickle.dump(degree,open(outRoot + "_degree_weighted.pkl","w"))
	for u in users:
		p_map_distance = createPMapDistance4Dest(users, edges, u)
		communities = createPMapCommunities4Dest(users, edges, u)
		pickle.dump(p_map_distance,open(outRoot + "_distance_weighted_" + str(u) + ".pkl","w"))
		pickle.dump(communities,open(outRoot + "_community_weighted_" + str(u) + ".pkl","w"))


def createPMapsP2P(uName,eName,outName):
	users = pickle.load(open(uName,"r"))
	edges = pickle.load(open(eName,"r"))
	pickle.dump(createPMapRandom(users,edges),open(outName + "_equal_weighted.pkl","w"))
	pickle.dump(createPMapDistance(users,edges),open(outName + "_distance_weighted.pkl","w"))
	pickle.dump(createPMapCommunities(users,edges),open(outName + "_community_weighted.pkl","w"))
	pickle.dump(createPMapDegree(users,edges),open(outName + "_degree_weighted.pkl","w"))
