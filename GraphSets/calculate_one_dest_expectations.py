import sys
sys.path.append("..")

from core.connection import Edge
from core.user import User
from expectation_calculation.calculate_expectations import calculate_expectations_very_pickled_1Dest

import cPickle as pickle
import time

rootN = "full_random/NODES"
rootE = "full_random/EDGES"
rootP = "full_random/pmaps_for_dests/PMAP"
rootO = "full_random/results_singular/FINAL"

avgsPer = [[0,0,0,0] for i in range(10)]
avgsSub = [[0,0,0,0] for i in range(10)]

num = 10

stime = time.time()
print 0

for seed in range(100,100+num):
	print "STARTING SEED " + str(seed)
	for i in range(1000,11000,1000):
		a = calculate_expectations_very_pickled_1Dest(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", "random", rootP + str(i), rootO + str(i) + "_equal_weighted_" + str(seed) + ".pkl", seed)
		#print "finished equal"
		b = calculate_expectations_very_pickled_1Dest(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", "distance", rootP + str(i), rootO + str(i) + "_distance_weighted_" + str(seed) + ".pkl", seed)
		#print "finished distance"
		c = calculate_expectations_very_pickled_1Dest(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", "communities", rootP + str(i), rootO + str(i) + "_community_weighted_.pkl", seed)
		#print "finished communities"
		d = calculate_expectations_very_pickled_1Dest(rootN + str(i) + ".pkl", rootE + str(i) + ".pkl", "degree", rootP + str(i), rootO + str(i) + "_degree_weighted.pkl", seed)
		#print "finished degree"
		avgsPer[i / 1000 - 1][0] += a["percentage"]
		avgsPer[i / 1000 - 1][1] += b["percentage"]
		avgsPer[i / 1000 - 1][2] += c["percentage"]
		avgsPer[i / 1000 - 1][3] += d["percentage"]
		avgsSub[i / 1000 - 1][0] += a["subtraction"]
		avgsSub[i / 1000 - 1][1] += b["subtraction"]
		avgsSub[i / 1000 - 1][2] += c["subtraction"]
		avgsSub[i / 1000 - 1][3] += d["subtraction"]
		print ">---FINISHING GRAPH OF SIZE " + str(i)
	print stime-time.time()

map(lambda x: map(lambda y: y/num, x), avgsPer)
map(lambda x: map(lambda y: y/num, x), avgsSub)

pickle.dump({"p":avgsPer, "s":avgsSub}, open(rootO + "10MORE.pkl", "w"))