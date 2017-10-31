from core.user import User

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


def set_communities_help(node, comms, c):
	if node.uid in comms[c]:
		yield c

def set_communities(nodes, comms):
	for node in nodes:
		nodes[node].comm = map(lambda c: set_communities_help(node, comms, c), range(len(comms)))