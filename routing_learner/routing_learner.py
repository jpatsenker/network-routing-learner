import math
import random
import sys
import copy
sys.path.append("..")

__author__ = 'jpatsenker'

from core.connection import distance
from learner.learner import LogisticRegressionLearner


def create_pmap_equal_weighted(graph):
	pmap = {}
	for dest in graph:
		pmap[dest] = {}
		for source in graph:
			pmap[dest][source] = {}
			for step in graph[source].friends:
				pmap[dest][source][step] = 1.0/float(graph[source].deg1)
	return pmap

def get_expected_paths(graph, pmap = None, sp = None):
	import time
	t = time.time()
	if not sp:
		sp = get_shortest_paths(graph)
	print "CALCULATED SHORTEST PATHS"
	print time.time()-t
	if not pmap:
		pmap = create_pmap_equal_weighted(graph)
	print "CALUCLATED PROBABILITY MAP"
	print time.time()-t
	ep = dict(sp)

	for i in range(10):
		np = copy.deepcopy(ep)
		for destination in graph:
			for source in graph:
				if destination in graph[source].friends:
					np[destination][source] = 1.0
				else:
					s = 0
					for step in graph[source].friends:
						s += pmap[destination][source][step] * float(ep[destination][step])
					#print s, ep[destination][source]
					np[destination][source] = s + 1
		ep = np
	print "Completed EP"
	print time.time()-t

	return ep




def get_shortest_paths(graph):
		#import time
		#t = time.time()
		sp = {}
		visited_zeros = {}
		for i in graph.keys():
			visited_zeros[i] = 0
		for destination in graph:
			sp[destination] = {}
			bfs_q = [destination]
			last_in_layer = destination
			curr_layer = 0
			visited = dict(visited_zeros)
			visited[destination] = 1
			while len(bfs_q) > 0:
				curr_id = bfs_q[0]
				bfs_q = bfs_q[1:]
				sp[destination][curr_id] = curr_layer
				friend = None
				#print len(graph[curr_id].friends)
				for friend in graph[curr_id].friends:
					if visited[friend] == 0:
						#print "adding " + friend + ", destination: " + destination
						bfs_q.append(friend)
						visited[friend] = 1
				if len(bfs_q) > 0 and curr_id == last_in_layer:
					last_in_layer = bfs_q[-1]
					curr_layer+=1
			#print "Destination finished..."
			#print time.time() - t
		print "Completed SP"
		return sp

class RoutingLearner:

	SAMPLE_SIZE = 100000

	def __init__(self, train_graph, test_graph, sptrain = None):
		self.train_graph = train_graph
		self.test_graph = test_graph
		self.shortest_paths_train = {}
		self.shortest_paths_test = {}
		if not sptrain:
			self.shortest_paths_train = get_shortest_paths(self.train_graph)
		else:
			self.shortest_paths_train = sptrain
		#self.shortest_paths_test = get_shortest_paths(self.test_graph)
		self.l = None
		self.norm_mean = None
		self.norm_min = None


	def normalize(self, feature_matrix):
		"""
		Normalize the feature matrix for training, store the normal mean & normal min
		:param feature_matrix: RECTANGULAR LIST OF LISTS
		:return: Normalized Matrix (-1 -> 1)
		"""
		if len(feature_matrix) > 0:
			nmin = [1000000 for _ in range(len(feature_matrix[0]))]
			nsum = [0 for _ in range(len(feature_matrix[0]))]
			for r in feature_matrix:
				for c in range(len(r)):
					nmin[c] = min(nmin[c], r[c])
					nsum[c] += r[c]
			self.norm_mean = map(lambda x: float(x)/float(len(feature_matrix)), nsum)
			self.norm_min = nmin
			return self.apply_normal(feature_matrix)
		else:
			return None

	def apply_normal_row(self, row):
		return map(lambda x: (float(row[x]-self.norm_min[x]))/float(self.norm_mean[x]), range(len(row)))

	def apply_normal(self, matrix):
		return map(self.apply_normal_row, matrix)

	def force_weights(self, weights):
		self.l.weights = weights

	def train(self):
		instancesGood = self.sample_data_plusone(RoutingLearner.SAMPLE_SIZE, self.train_graph, self.shortest_paths_train)
		instancesBad = self.sample_data_minusone(RoutingLearner.SAMPLE_SIZE, self.train_graph, self.shortest_paths_train)
		feature_matrix_g = self.extract_features(instancesGood)
		feature_matrix_b = self.extract_features(instancesBad)
		print len(feature_matrix_b)
		print len(feature_matrix_g)
		feature_matrix_g.extend(feature_matrix_b)
		feature_matrix = feature_matrix_g
		print len(feature_matrix)
		normalized_feature_matrix = self.normalize(feature_matrix)
		ys = [1 for _ in range(RoutingLearner.SAMPLE_SIZE)]
		ys.extend([-1 for _ in range(RoutingLearner.SAMPLE_SIZE)])
		print len(ys)
		zs = self.perform_transform(normalized_feature_matrix)
		self.l = LogisticRegressionLearner(zs,ys)
		self.l.learn()
		print self.l.weights

	# def test(self, test_data, test_target):
	# 	test_data_instances = []
	# 	for i in test_data:
	# 		for j in test_data:
	# 			test_data_instances.append({"source":test_data[i], "destination":test_data[j]})
	#
	# 	for test_point in test_data_instances:
	# 		r = dict(self.predict(test_point["source"], test_point["destination"], test_data))
	# 		choice = r - max(r.values())
	# 		error = test_target[test_point["destination"].uuid][test_point["source"].uuid]

	def test(self, test_data, test_target, eps = None, sps = None):
		if not sps:
			sps = get_shortest_paths(copy.deepcopy(test_data))
		if not eps:
			eps = self.get_expected_paths_from_weights(copy.deepcopy(test_data), sp = sps)

		#print test_target
		stretch = 0
		stretch_sum = 0
		count = 0
		for destination in test_target:
			for source in test_target[destination]:
				if source is not destination and source not in test_data[destination].friends:
					stretch += float(eps[destination][source])/float(test_target[destination][source])
					if eps[destination][source] < test_target[destination][source]:
						print destination, source, eps[destination][source], test_target[destination][source]
					stretch_sum += eps[destination][source] - test_target[destination][source]
					count += 1
		print stretch
		print stretch_sum
		stretch /= float(count)
		stretch_sum /= float(count)
		return stretch, stretch_sum


	def get_expected_paths_from_weights(self, graph, sp = None):
		import time
		t = time.time()
		if not sp:
			sp = get_shortest_paths(graph)
		print "CALCULATED SHORTEST PATHS"
		print time.time()-t
		ep = copy.deepcopy(sp)

		for i in range(10):
			#print "NEXT ITER"
			np = copy.deepcopy(ep)
			for destination in graph:
				for source in graph:
					#print "--NEXT SOURCE"
					s = 0
					ps = dict(self.predict(graph[source], graph[destination], graph))
					#print "AS:FLKJDS," , ps.values()
					a = ps.values()
					ps = dict(zip(ps.keys(), map(lambda p: p/float(sum(a)), a)))
					#sanity_check = sum(ps.values())
					#epsilon = .0000001 #for sanity check
					# if abs(sanity_check - 1.0)>epsilon:
					# 	print "SANITY CHECK FAILED!!!", sanity_check-1.0
					# 	exit(1)
					if destination in graph[source].friends:
						np[destination][source] = 1.0
					else:
						for step in graph[source].friends:
							s += float(ps[step]) * float(ep[destination][step])
							#print float(ps[step]), float(ep[destination][step]), float(ep[destination][source]), float(sp[destination][step]), float(sp[destination][source])
						# if s<ep[destination][source]-1:
						# 	print s, ep[destination][source], destination, source, ps, map(lambda x: ep[destination][x], ps.keys())
						np[destination][source] = float(s + 1)
					# if np[destination][source] - ep[destination][source] < -epsilon:
					# 	print "SANITY CHECK FAILED!!!", destination, source, np[destination][source], ep[destination][source]
					# 	exit(1)
					#print s
			ep = np
		print "Completed EP"
		print time.time()-t
		return ep


	def predict(self, node, destination, ref_graph):
		#print len(node.friends)
		xs_features = map(lambda x: self.extract_features([{"source":node, "destination": destination, "step": ref_graph[x]}])[0], node.friends)
		#print xs_features
		normalized_features = self.apply_normal(xs_features)
		#print normalized_features
		return zip(node.friends, map(lambda x: self.l.hypfunc_log(x), normalized_features))
		#return zip(node.friends, map(lambda x: random.random(), xs_features))

	def sample_data_plusone_old(self, size, graph,spaths):
		instances = []
		for n in range(size):
			#print "Next Plus One Sample"
			r1 = random.random()*len(graph)
			r2 = random.random()*len(graph)
			u1 = graph.keys()[int(r1)]
			u2 = graph.keys()[int(r2)]

			best_choice = None
			least_hops = 1000
			for f in graph[u1].friends:
				#print spaths
				if spaths[u2][f] < least_hops:
					best_choice = f
					least_hops = spaths[u2][f]

			instances.append({"source":graph[u1], "destination":graph[u2], "step":graph[best_choice]})
		return instances

	def sample_data_plusone(self, size, graph,spaths):
		instances = []
		for n in range(size):
			#print "Next Plus One Sample"
			r1 = random.random()*len(graph)
			r2 = random.random()*len(graph)
			u1 = graph.keys()[int(r1)]
			u2 = graph.keys()[int(r2)]

			best_choice = None
			least_hops = 1000
			for f in graph[u1].friends:
				#print spaths
				if spaths[u2][f] < least_hops:
					best_choice = f
					least_hops = spaths[u2][f]

			instances.append({"source":graph[u1], "destination":graph[u2], "step":graph[best_choice]})
		return instances


	def get_choices(self, graph, spaths):
		r1 = random.random()*len(graph)
		r2 = random.random()*len(graph)
		u1 = graph.keys()[int(r1)]
		u2 = graph.keys()[int(r2)]

		best_choice = None
		least_hops = 1000
		for f in graph[u1].friends:
			#print u2, f
			#print spaths
			if spaths[u2][f] < least_hops:
				best_choice = f
				least_hops = spaths[u2][f]

		choices = list(graph[u1].friends)
		choices.remove(best_choice)
		return u1, u2, best_choice, choices



	def sample_data_minusone(self, size, graph, spaths):
		instances = []
		for n in range(size):
			#print "Next Minus One Sample"
			source, destination, best_choice, choices = self.get_choices(graph, spaths)
			while len(choices) == 0:
				source, destination, best_choice, choices = self.get_choices(graph, spaths)

			r3 = random.random()*len(choices)
			u3 = graph[choices[int(r3)]]

			instances.append({"source":graph[source], "destination":graph[destination], "step":u3})
		return instances

	def extract_features(self, instances):
		print "METHOD NOT IMPLEMENTED!!!"
		return []

	def perform_transform(self, features):
		print "METHOD NOT IMPLEMENTED!!!"
		return []

class RoutingLearnerFeatureSetOne(RoutingLearner, object):

	def __init__(self, train_graph, test_graph, target=None):
		super(RoutingLearnerFeatureSetOne, self).__init__(train_graph, test_graph, target)

	def perform_transform(self, features):
		return features

	def extract_features(self, instances):
		feature_transform = []
		for i in instances:
			source = i["source"]
			destination = i["destination"]
			step = i["step"]

			dist = distance(step.pos, destination.pos)
			communities_in_common = len(set.intersection(set(step.comm),set(destination.comm)))
			degree = step.deg1
			a = .001
			t = 275
			dist_weighted = math.tanh(a*(dist - t))
			log_degree = math.log(degree)

			#DISTANCE should be 1/(dist+epsilon)
			#Weighted Dist should be 1/(wd+epsilon)

			feature_transform.append([1, dist, communities_in_common, degree, dist_weighted, log_degree])
		return feature_transform


import cPickle as pickle
import random_walk.random_walk as rw

graph5000 = pickle.load(open("/Users/jpatsenker/gowalla_research/GraphSets/full_random/NODES10000.pkl", "r"))
graph6000 = pickle.load(open("/Users/jpatsenker/gowalla_research/GraphSets/full_random/NODES2000.pkl", "r"))


train = rw.doRandomWalk(graph5000, 1000)
test = train
#test = rw.doRandomWalk(graph5000, 1000)
print "Made graphs"

spaths_test = get_shortest_paths(test)

#print spaths_test
#print test
# for dest in spaths_test:
# 	for node in spaths_test[dest]:
# 		#print node, "clear"
# 		for step in test[node].friends:
# 			try:
# 				assert abs(spaths_test[dest][node] - spaths_test[dest][step]) < 1.2
# 			except:
# 				print node, step, spaths_test[dest][node], spaths_test[dest][step]
# 				assert 1==0
# print "GOOD"


epaths_train = get_expected_paths(train)



#r = RoutingLearnerFeatureSetOne(train, None)
q = RoutingLearnerFeatureSetOne(train, None, target = epaths_train)
#r.train()
q.train()
#q.force_weights([0,0,0,0,1,0])
print q.l.weights


# spaths = get_shortest_paths(train)
# epaths = get_expected_paths(train, spaths)

# epaths_test = get_expected_paths(test, spaths_test)


#print r.predict(graph5000[graph5000.keys()[0]], graph5000[graph5000.keys()[20]], graph5000)

stretch, stretch_sum = q.test(test, spaths_test, eps=None, sps=spaths_test)

print stretch, stretch_sum


###LEARNED ON 9000: [-0.06951276 -0.00234584  0.33283665  0.00193841  0.01865547]


###2000 -> 1000 [ 0.         -0.44630104  0.02742932  0.0133496   0.02947272], 3.3510509344 8.39544500543


###Made graphs
# Completed SP
# Completed SP
# CALCULATED SHORTEST PATHS
# 377.604709148
# CALUCLATED PROBABILITY MAP
# 512.431463957
# Completed EP
# 4001.32559705
# 30
# 30
# 60
# 60
# [ 0.         -0.33183169 -0.10200842  1.07628603  0.02375747]
# CALCULATED SHORTEST PATHS
# 2.09808349609e-05
# Completed EP
# 1014.12992692
# 142720 42072 2.0 2
# 142720 2334055 2.0 2
# 134002 42072 2.0 2
# 134002 54611 2.0 2
# 30803 310105 2.0 2
# 30803 157234 2.0 2
# 19961 28076 2.0 2
# 5547 54611 2.0 2
# 2373261 54775 2.0 2
# 151398 243945 2.0 2
# 101630 104414 2.0 2
# 2171565 42072 2.0 2
# 2171565 2117458 2.0 2
# 10677 42328 2.0 2
# 3282821.42592
# 8655542.27344
# 3.29692567395 8.69272977141