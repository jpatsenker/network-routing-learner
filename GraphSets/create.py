import sys
sys.path.append("../..")

from gowalla_research.random_walk.random_walk import doRandomWalkPickle2Pickle
from gowalla_research.core.user import User
from gowalla_research.core.connection import Edge


for i in range(10):
	doRandomWalkPickle2Pickle("nodes_data_dictionary.pkl", "full_random/NODES" + str(i+1) + "000.pkl",(i+1)*1000)
