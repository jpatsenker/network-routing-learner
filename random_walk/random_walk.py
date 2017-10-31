import sys
sys.path.append("..")

from core.connection import Edge
from core.user import User
import cPickle as pickle
import random


def selectRandomUser(users):
		return users.keys()[int(random.random()*len(users.keys()))]

def doRandomWalk(users, size):
	availableMovements = set()
	randomSet = set()
	edgeSet = {}
	u1 = selectRandomUser(users)
	randomSet.add(u1)
	availableMovements = availableMovements.union(set(users[u1].friends))
	for i in range(size):
		#print availableMovements
			uNext = random.sample(availableMovements, 1)[0]
			randomSet.add(uNext)
			availableMovements = availableMovements.union(set(users[uNext].friends))
			availableMovements.remove(uNext)
			availableMovements = availableMovements.difference(set(randomSet))
	popSet = {}
	for r in randomSet:
			friendsOfR = set(users[r].friends).intersection(randomSet)
			commOfR = users[r].comm
			posOfR = users[r].pos
			popSet[r] = User(r,commOfR,posOfR,friendsOfR)
	return popSet

def doRandomWalkPickle2Pickle(pickleStart,pickleDestination,size):
	users = pickle.load(open(pickleStart,"r"))
	popSet = doRandomWalk(users,size)
	pickle.dump(popSet, open(pickleDestination,"w"))
