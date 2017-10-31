import sys
sys.path.append("../..")

from gowalla_research.core.connection import Edge
from gowalla_research.core.user import User
from gowalla_research.expectation_calculation.create_edge_table import create_edges_dictionary_P2P

testF = "full_random/NODES"
testO = "full_random/EDGES"

for i in range(10):
	create_edges_dictionary_P2P(testF + str((i+1)*1000) + ".pkl",testO + str((i+1)*1000) + ".pkl")
