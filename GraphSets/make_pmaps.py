import sys
sys.path.append("..")

from core.connection import Edge
from core.user import User
from expectation_calculation.createProbabilityMap import createPMapsP2PE
from expectation_calculation.createProbabilityMap import createPMapDistanceDatabase
import cPickle as pickle

rootU = "full_random/NODES"
rootE = "full_random/EDGES"
rootO = "full_random/pmaps_for_dests/PMAP"

# pmaps = []
# for i in range(1000,11000,1000):
# 	createPMapsP2PE(rootU + str(i) + ".pkl", rootE + str(i) + ".pkl", rootO + str(i))
# 	print "created for " + str(i)

import mysql.connector

cnx = mysql.connector.connect(user='root', password='password',
                              host='127.0.0.1',
                              database='gowalla_pmaps')

database = cnx.cursor()

for i in range(1000,2000,1000):
	users = pickle.load(open(rootU + str(i) + ".pkl","r"))
	edges = pickle.load(open(rootE + str(i) + ".pkl","r"))
	createPMapDistanceDatabase(users,edges,cnx,str(i))

database.close()
cnx.close()