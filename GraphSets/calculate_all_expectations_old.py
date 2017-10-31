import sys
sys.path.append("../..")

from gowalla_research.core.connection import Edge
from gowalla_research.core.user import User
from gowalla_research.expectation_calculation.calculate_expectations import calculate_expectations


rootN = "full_random/NODES"
rootE = "full_random/EDGES"
rootP = "full_random/PMAP"
rootO = "full_random/FINAL"

for i in range(1000,11000,1000):
	calculate_expectations(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", rootP + str(i) + ".pkl_equal_weighted.pkl", rootO + str(i) + "_equal_weighted.pkl")
	print "finished equal"
	calculate_expectations(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", rootP + str(i) + ".pkl_distance_weighted.pkl", rootO + str(i) + "_distance_weighted.pkl")
	print "finished distance"
	calculate_expectations(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", rootP + str(i) + ".pkl_community_weighted.pkl", rootO + str(i) + "_community_weighted.pkl")
	print "FINISHING GRAPH OF SIZE " + str(i)
