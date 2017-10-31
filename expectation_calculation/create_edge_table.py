import sys
sys.path.append("../..")

from gowalla_research.core.connection import Edge
from gowalla_research.core.user import User
import cPickle as pickle

def create_edges_dictionary_P2P(inpName, outName):
	users = pickle.load(open(inpName,"r"))

	connections = {}

	eid_count = 0

	for u in users:
		connections[u] = {}
		for f in users[u].friends:
			connections[u][f] = Edge(eid_count, u,f)
			connections[u][f].fill_in_data(users)
			eid_count+=1

	pickle.dump(connections,open(outName,"w"))
