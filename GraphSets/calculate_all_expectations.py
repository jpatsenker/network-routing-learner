import sys
sys.path.append("../.")

from core.connection import Edge
from core.user import User
from expectation_calculation.calculate_expectations import calculate_expectations_very_pickled
#from make_pmaps import pmaps

rootN = "full_random/NODES"
rootE = "full_random/EDGES"
rootP = "full_random/pmaps_for_dests/PMAP"
rootO = "full_random/results_two/FINAL"



for i in range(1000, 10000, 1000):
	ind = i / 1000 - 1
	calculate_expectations_very_pickled(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", "random", rootP + str(i), rootO + str(i) + "_equal_weighted.pkl")
	print "finished equal"
	calculate_expectations_very_pickled(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", "distance", rootP + str(i), rootO + str(i) + "_distance_weighted.pkl")
	print "finished distance"
	calculate_expectations_very_pickled(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", "communities", rootP + str(i), rootO + str(i) + "_community_weighted.pkl")
	print "finished communities"
	calculate_expectations_very_pickled(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", "degree", rootP + str(i), rootO + str(i) + "_degree_weighted.pkl")
	print "finished degree"
	print "FINISHING GRAPH OF SIZE " + str(i)
