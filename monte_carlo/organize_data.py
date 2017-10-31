import cPickle as pickle
from cPickle import Unpickler
import math
import random
from numpy.random import choice

class User:
	
	def __init__(self,uid,comm,pos,friends):
		self.uid=uid
		self.comm=comm
		self.pos=pos
		self.friends=friends
		self.deg1=len(friends)
		self.deg2=None
	
def loadCommunities(filename):
	comms = []
	with open(filename, "r") as communities:
		for line in communities:
			community = line.split(" ")[:-1]
			communityN = []
			for c in community:
				#try:
				communityN.append(str(c))
				#except:
					#print "Excluding " + str(c) + " (non numeric)"
			comms.append(communityN)
	return comms

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

class Edge:
	
	def __init__(self, eid, uid1, uid2):
		self.eid = eid
		self.uid1 = uid1
		self.uid2 = uid2

	def fill_in_data(self,users):
		u1= users[self.uid1]
		u2 = users[self.uid2]
		self.dist = distance(u1.pos, u2.pos)
		self.mutual = len(set.intersection(set(u1.friends),set(u2.friends)))
		self.commoncom = len(set.intersection(set(u1.comm),set(u2.comm)))
		self.rp1 = u2.deg1
		self.rp2 = u2.deg2

	def __str__(self):
		return str(self.uid1) + "---" + str(self.uid2)

	def __repr__(self):
		return self.__str__()

def loadLocations(filename):
	u = Unpickler(open(filename, "r"))
	locationsDict = u.load()
	return locationsDict

def loadFriends(filename):
	u = Unpickler(open(filename, "r"))
	friendsDict = u.load()
	return friendsDict

def createUserDict(locDict, friendsDict, comms):
	users = {} #uid -> user-obj
	ukeys = set.union(set(locDict.keys()),set(friendsDict.keys()))
	for u in ukeys:
		loc = None
		friends = []
		if u in locDict.keys():
			loc = locDict[u]
		if u in friendsDict.keys():
			friends.extend(map(str,friendsDict[u]))
		users[str(u)] = User(str(u),[],loc,friends)
	for uid in users:
		user = users[uid]
		fs = user.friends
		user.deg2 = sum(map(lambda x: users[x].deg1, fs))
	communityNum = 0
	for community in comms:
		#if communityNum == 1:
		#	print users.keys()[0]
		#	print community[0]
		#	print type(users.keys()[0]) is int
		#	print type(community[0]) is int
		#	print users.keys()[0] == community[0]
		for uid in community:
			if str(uid) in users.keys():
				#print "adding comm"
				users[uid].comm.append(communityNum)
		communityNum += 1
	return users

	
#locationsDict = loadLocations("../Data/gowalla_users_locations_USonly.pck")

#friendsDict = loadFriends("../Data/gowalla_spatial_network_USonly.pck")

#comms = loadCommunities("../Data/SLPAw_gowalla_spatial_network_run1_r0.2_v4_TTL100_T100.icpm")
#comms = loadCommunities("../Data/SLPAw_gowalla_spatial_network_run1_r0.5_v4_TTL100_T100.icpm")

#print "loaded data, creating User Dictionary"

#users = createUserDict(locationsDict, friendsDict, comms)

#pickle.dump(users, open("nodes_data_dictionary_full.pkl","wb"))

def selectRandomUser(users):
	return users.keys()[int(random.random()*len(users.keys()))]

def run_monte_carlo_single(users, chooseFunc):
	sid = selectRandomUser(users)
	#did = users[users[sid].friends[0]].friends[2]
	did = selectRandomUser(users)
	print "START: " + str(sid)
	print "--des: " + str(did)
	cid = sid
	iter_count = 0
	while cid is not did:
		if(iter_count == 100):
			break
		edges = []
		count = 0
		for f in users[cid].friends:
			e = Edge(count,cid,f)
			e.fill_in_data(users)
			edges.append(e)
			count += 1
		cid = chooseFunc(edges, users, did)
		#print "Out of edges, " + str(map(lambda x: x.uid2, edges))
		print "SENDING TO " + str(cid)
		#print "-destination" + str(did)
		iter_count += 1
	print "START: " + str(sid)
        print "--des: " + str(did)
	return iter_count


def choose_by_routingpower(edges, users, did):
	rps = {}
	for e in edges:
		rps[e] = e.rp1
	r = random.random()*sum(rps.values())
	s = 0
	q = None
	for rp in rps:
		q = rp
		if s>=r:
			return rp.uid2
		s+=rps[rp]
	return rp.uid2	

def choose_by_commoncom(edges, users, did):
	best = {}
	for e in edges:
		best[e] = len(set.intersection(set(users[e.uid2].comm),set(users[did].comm)))
	r = random.random()*sum(best.values())
	s=0
	for b in best:
		s+=best[b]
		if s>=r:
			print best[b]
			return b.uid2
	

def choose_by_closeness_no_back(edges, users, did, memory):
	distancesByEdge = {}
        for e in edges:
                if e.uid2 is did:
                        return did
		if e.uid2 in memory:
			continue
		else:
                	distancesByEdge[e] = 1.0/float(distance(users[e.uid2].pos,users[did].pos)+1)
        total = sum(distancesByEdge.values())
        print distancesByEdge
        r = random.random()*total
        i=0
        s=0
        print r, total
	if len(distancesByEdge.keys()) is 0:
		return edges[0].uid1
        d = distancesByEdge.keys()[i]
	while s<=r:
                i+=1
                if i>=len(distancesByEdge.keys()):
                        memory.append(d.uid2)
			return d.uid2
                d = distancesByEdge.keys()[i]
                s += distancesByEdge[d]
	memory.append(d.uid2)
        return d.uid2


def choose_by_closeness(edges, users, did):
	distancesByEdge = {}
	for e in edges:
		if e.uid2 is did:
			return did
		distancesByEdge[e] = 1.0/float(distance(users[e.uid2].pos,users[did].pos)+1)
	total = sum(distancesByEdge.values())
	#print distancesByEdge

	c= choice(distancesByEdge.keys(),1,map(lambda x: x/total, distancesByEdge.values()))
	#print c
	return c[0].uid2

	
	r = random.random()*total
	i=0
	s=0
	#print r, total
	d = distancesByEdge.keys()[i]
	while s<=r:
		i+=1
		if i>=len(distancesByEdge.keys()):
			return d.uid2
		d = distancesByEdge.keys()[i]
		s += distancesByEdge[d]
	return d.uid2

def choose_by_closeness_weighted(edges,users,did):
	distancesByEdge = {}
        for e in edges:
                if e.uid2 is did:
                        return did
                distancesByEdge[e] = 1.0/float(distance(users[e.uid2].pos,users[did].pos)+1)
        total = sum(distancesByEdge.values())
        #print distancesByEdge
        r = (random.random()**2)*total
        i=0
        s=0
        #print r, total
        d = distancesByEdge.keys()[i]
        while s<=r:
                i+=1
                if i>=len(distancesByEdge.keys()):
                        return d.uid2
                d = distancesByEdge.keys()[i]
                s += distancesByEdge[d]
        return d.uid2




users = pickle.load(open("nodes_data_dictionary_full.pkl","r"))

d = []

res = []

for i in range(1000):
	res.append(run_monte_carlo_single(users, choose_by_closeness))

print res
print float(sum(res))/float(len(res))


#print run_monte_carlo_single(users,  choose_by_closeness)
#print run_monte_carlo_single(users, lambda x,y,z:  choose_by_closeness_no_back(x, y, z, d))






