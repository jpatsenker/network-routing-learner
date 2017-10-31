import numpy as np
from sklearn.linear_model import LogisticRegression
from core.user import User
import cPickle as pickle
import random
import copy
import math
import time

'''
import toolkit.tools as t
from core.user import User
i=100
j=5
test=100
n = t.load_dict()
ni = t.reindex_dict(n)
ws = t.learn_exp(ni, [i]*j)
ws[len(ws)] = sum(ws.values())/float(len(ws))
nt = t.random_walk(ni,test)
exps,shps = t.run_all_exp_comp(nt, ws[len(ws)-1])
str_unif = exps['uniform']/(shps+1.)
str_loc = exps['location']/(shps+1.)
str_comm = exps['common_communities']/(shps+1.)
str_wloc = exps['weighted_location']/(shps+1.)
str_deg = exps['degree']/(shps+1.)
str_w = exps['weighted']/(shps+1.)
'''


def tt(n):
	k = 0
	z = 0
	ts = time.time()
	for i in range(n):
		z = k + i
	return time.time() - ts


def tsq(n):
	k = 0
	z = 0
	ts = time.time()
	for i in range(n):
		for j in range(n):
			z = k + i + j
	return time.time() - ts


def map_tt(n):
	k = 0
	ts = time.time()
	map(lambda i: k + i, range(n))
	return time.time() - ts


def learn_shps(full, sizes):
	st = time.time()
	weights = {}
	for s in sizes:
		weights[s] = learn_inst(random_walk(full, s))
		print "FINISHED SIZE", s, "time: ", time.time() - st
	return weights


def learn_exp(full, sizes):
	st = time.time()
	weights = {}
	for s in range(len(sizes)):
		weights[s] = learn_inst_exp(random_walk(full, sizes[s]))
		print "FINISHED SIZE", s, "time: ", time.time() - st
	return weights


def learn_inst(nodes):
	inodes = reindex_dict(nodes)
	shps = shortest_paths(inodes)
	inst = extract_instances_full(inodes)
	obj = []
	for inst_x in inst:
		og = shps[inst_x['source'].uid, inst_x['destination'].uid]
		nu = shps[inst_x['step'].uid, inst_x['destination'].uid]
		if og == nu + 1:
			obj.append(1)
		else:
			obj.append(0)
	fm = extract_feature_matrix(inst)
	fms = standardize(fm)
	return logistic_regression(fms, obj)


def learn_inst_exp(nodes):
	inodes = reindex_dict(nodes)
	shps = shortest_paths(inodes)
	inst = extract_instances_full(inodes)
	exp = get_expectations(inodes, shps)
	obj = []
	for inst_x in inst:
		og = exp[inst_x['source'].uid, inst_x['destination'].uid]
		nu = exp[inst_x['step'].uid, inst_x['destination'].uid]
		if og - nu > 0.25:
			obj.append(1)
		else:
			obj.append(0)
	fm = extract_feature_matrix(inst)
	fms = standardize(fm)
	return logistic_regression(fms, obj)


def test_weights(nodes, weights):
	inodes = reindex_dict(nodes)
	shps = shortest_paths(inodes)
	pmap = make_pmap(inodes, lambda x, y, z: pmap_weights(x, y, z, weights))
	exps = get_expectations(inodes, shps, pmap)
	return exps, shps


def run_all_exp(full, sizes):
	st = time.time()
	exps = {}
	for s in sizes:
		exps[s] = run_def_exp(random_walk(full, s))
		print "FINISHED SIZE", s, "time: ", time.time() - st
	return exps


def run_def_exp(nodes):
	# log time
	st = time.time()

	# GET FULL REINDEX
	ins = reindex_dict(nodes)

	shps = shortest_paths(ins)
	print "FINISHED SHORTEST PATHS, time: ", time.time() - st

	up = make_pmap(ins, uniform)
	print "FINISHED UNIFORM PMAP, time: ", time.time() - st

	lp = make_pmap(ins, closer)
	print "FINISHED LOCATION PMAP, time: ", time.time() - st

	wlp = make_pmap(ins, weighted_closer)
	print "FINISHED WEIGHTED LOCATION PMAP, time: ", time.time() - st

	exp_u = get_expectations(ins, shps, up)
	print "FINISHED UNIFORM EXPECTATION, time: ", time.time() - st

	exp_lp = get_expectations(ins, shps, lp)
	print "FINISHED LOCATION EXPECTATION, time: ", time.time() - st

	exp_wlp = get_expectations(ins, shps, wlp)
	print "FINISHED WEIGHTED LOCATION EXPECTATION, time: ", time.time() - st

	return {'uniform': exp_u, 'location': exp_lp, 'weighted_location': exp_wlp}, shps


def run_all_exp_comp(full, sizes, weights):
	st = time.time()
	exps = {}
	shps = {}
	for s in sizes:
		exps[s], shps[s] = runDefExpComp(random_walk(full, s), weights)
		print "FINISHED SIZE", s, "time: ", time.time() - st
	return exps, shps


def run_all_exp_comp_ws(nodes, weights):
	st = time.time()
	exps = {}
	shps = {}
	for w in range(len(weights)):
		exps[w], shps[w] = runDefExpComp(nodes, weights[w])
		print "FINISHED WEIGHT", w, "time: ", time.time() - st
	return exps, shps


def runDefExpComp(nodes, weights):
	# log time
	st = time.time()

	# GET FULL REINDEX
	ins = reindex_dict(nodes)

	shps = shortest_paths(ins)
	print "FINISHED SHORTEST PATHS, time: ", time.time() - st

	up = make_pmap(ins, uniform)
	print "FINISHED UNIFORM PMAP, time: ", time.time() - st

	lp = make_pmap(ins, closer)
	print "FINISHED LOCATION PMAP, time: ", time.time() - st

	wlp = make_pmap(ins, weighted_closer)
	print "FINISHED WEIGHTED LOCATION PMAP, time: ", time.time() - st

	ccp = make_pmap(ins, communities)
	print "FINISHED COMMON COMMUNITIES PMAP, time: ", time.time() - st

	pp = make_pmap(ins, popular)
	print "FINISHED DEGREE PMAP, time: ", time.time() - st

	wp = make_pmap(ins, lambda x, y, z: pmap_weights(x, y, z, weights=weights))
	print "FINISHED WEIGHTED PMAP, time: ", time.time() - st

	exp_u = get_expectations(ins, shps, up)
	print "FINISHED UNIFORM EXPECTATION, time: ", time.time() - st

	exp_lp = get_expectations(ins, shps, lp)
	print "FINISHED LOCATION EXPECTATION, time: ", time.time() - st

	exp_wlp = get_expectations(ins, shps, wlp)
	print "FINISHED WEIGHTED LOCATION EXPECTATION, time: ", time.time() - st

	exp_ccp = get_expectations(ins, shps, ccp)
	print "FINISHED COMMON COMMUNITIES EXPECTATION, time: ", time.time() - st

	exp_pp = get_expectations(ins, shps, pp)
	print "FINISHED DEGREE EXPECTATION, time: ", time.time() - st

	exp_wp = get_expectations(ins, shps, wp)
	print "FINISHED WEIGHTED EXPECTATION, time: ", time.time() - st

	return {'uniform': exp_u, 'location': exp_lp, 'weighted_location': exp_wlp, 'common_communities': exp_ccp, 'degree': exp_pp, 'weighted': exp_wp}, shps


def load_dict():
	nodes = pickle.load(open("GraphSets/nodes_data_dictionary.pkl", 'r'))
	return nodes


def reindex_dict(nodes):
	inodes = dict()
	ndic = dict(zip(range(len(nodes.keys())), nodes.keys()))
	rev = {v: k for k, v in ndic.iteritems()}
	for n in range(len(nodes)):
		inodes[n] = copy.copy(nodes[ndic[n]])
		inodes[n].uid = n
		if n % 1000 == 0:
			print "Done: ", n
	for n in inodes:
		inodes[n].friends = map(lambda f: rev[f], inodes[n].friends)
		if n % 1000 == 0:
			print "Done: ", n
	return inodes


def make_adjacency_list(f):
	a = np.zeros([len(f), len(f)])
	for i in range(len(f)):
		for n in f[i].friends:
			a[i, n] = 1
		if i % 100 == 0:
			print "Done " + str(i)
	return a


def shortest_path(nodes, d):
	shps = np.array(np.zeros([len(nodes)]) - 1)
	queue = [d]
	vstd = np.zeros([len(nodes)])
	vstd[d] = 1
	count = 0
	l = d
	while len(queue) > 0:
		c = queue.pop(0)
		fr = nodes[c].friends
		for f in fr:
			if vstd[f] != 1:
				queue.append(f)
				vstd[f] = 1
		shps[c] = count
		if l == c and len(queue) > 0:
			l = queue[-1]
			count += 1
	return shps


def shortest_paths(nodes):
	shps = np.zeros([len(nodes), len(nodes)]) - 1
	for d in nodes:
		shps[d] = shortest_path(nodes, d)
		if d % 100 == 0:
			print "Done " + str(d)
	return shps


def shortest_paths_n_dest(nodes, use=None):
	if not use:
		use = [0]
	shps = np.zeros([len(nodes), len(nodes)]) - 1
	for d in range(len(use)):
		shps[d] = shortest_path(nodes, use[d])
		if d % 100 == 0:
			print "Done " + str(d)
	return shps


# def shortestPaths(nodes):
# 	shps = np.zeros([len(nodes),len(nodes)])-1
#
# 	for d in range(len(nodes.keys())):
# 		queue = [d]
# 		vstd = np.zeros([len(nodes)])
# 		count = 0
# 		l = d
# 		while len(queue)>0:
# 			c = queue.pop(0)
# 			fr = nodes[c].friends
# 			for f in fr:
# 				if vstd[f]!=1:
# 					queue.append(f)
# 			vstd[c]=1
# 			shps[c,d]=count
# 			if l == c and len(queue)>0:
# 				l = queue[-1]
# 				count+=1
# 		if d%100==0:
# 			print "Done " + str(d)
# 	return shps


def selectRandomUser(users):
	return users.keys()[int(random.random() * len(users.keys()))]


def random_walk(users, size):
	available = set()
	rset = set()
	u1 = selectRandomUser(users)
	rset.add(u1)
	available = available.union(set(users[u1].friends))
	for i in range(size - 1):
		if i % 1000 == 0:
			print i
		uNext = random.sample(available, 1)[0]
		rset.add(uNext)
		available = available.union(set(users[uNext].friends))
		available.remove(uNext)
		available = available.difference(set(rset))
	popSet = {}
	for r in rset:
		friendsOfR = set(users[r].friends).intersection(rset)
		commOfR = users[r].comm
		posOfR = users[r].pos
		popSet[r] = User(r, commOfR, posOfR, friendsOfR)
	return popSet


def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def uniform(frnds, nodes, dest):
	return [1 / float(len(frnds))] * len(frnds)


def closer(frnds, nodes, dest):
	dists = map(lambda n: distance(nodes[n].pos, nodes[dest].pos), frnds)
	ps = map(lambda d: 1 / (d + 1), dists)
	ps = np.array(ps)
	ps /= ps.sum()
	return ps


def popular(frnds, nodes, dest):
	popularity = map(lambda n: float(nodes[n].deg1), frnds)
	popularity = np.array(popularity)
	popularity /= popularity.sum()
	return popularity


def communities(frnds, nodes, dest):
	ccs = map(lambda step: len(set.intersection(set(nodes[step].comm), set(nodes[dest].comm))), frnds)
	ccs = np.array(ccs)
	ccs /= ccs.sum()
	return ccs


def pmap_weights(frnds, nodes, dest, weights=None):
	ps = []
	for f in frnds:
		feat = extract_feature({'source': None, 'destination': nodes[dest], 'step': nodes[f]})
		ps.append(np.dot(feat, np.array(weights)))
	ps = np.array(ps)
	ps /= ps.sum()
	return ps


def sig1(dist, a=0.001, t=275):
	return np.apply_along_axis(lambda d: a * (d - t), 0, dist)


def weighted_closer(frnds, nodes, dest, wfunc=sig1):
	dists = map(lambda n: distance(nodes[n].pos, nodes[dest].pos), frnds)
	dists = wfunc(dists)
	ps = map(lambda d: 1 / (d + 1), dists)
	ps = np.array(ps)
	ps /= ps.sum()
	return ps


def mean_vec(nar):
	return np.ones(nar.shape[0]).dot(nar) / nar.shape[0]


def center(nar):
	one_mat = np.ones(nar.shape[0]).reshape(nar.shape[0], 1)
	mvec = mean_vec(nar).reshape(nar.shape[1], 1).T
	return nar - np.dot(one_mat, mvec)


def standardize(nar):
	cnar = center(nar)
	std_vec = np.std(nar, axis=0)
	return cnar / std_vec


def make_pmap(nodes, r_alg=uniform):
	pmap = np.zeros([len(nodes), len(nodes), len(nodes)])
	for d in nodes:
		for n in nodes:
			fr = nodes[n].friends
			ps = r_alg(fr, nodes, d)
			for f in range(len(fr)):
				pmap[d, n, fr[f]] = ps[f]
		if d % 100 == 0:
			print "Done: ", d
	return pmap


def make_pmap_n_dest(nodes, use=None, r_alg=uniform):
	if not use:
		use = [0]
	pmap = np.zeros([len(use), len(nodes), len(nodes)])
	for d in range(len(use)):
		for n in nodes:
			fr = nodes[n].friends
			ps = r_alg(fr, nodes, use[d])
			for f in range(len(fr)):
				pmap[d, n, fr[f]] = ps[f]
		if d % 100 == 0:
			print "Done: ", d
	return pmap


def get_expectations_n_dest(nodes, use=[0], shps=None, pmap=None, iters=10):
	if shps is None:
		shps = shortest_paths_n_dest(nodes, use=use)
	if pmap is None:
		pmap = make_pmap_n_dest(nodes, use=use)
	expect = np.copy(shps)
	nexp = np.zeros(expect.shape)
	for i in range(iters):
		print "Starting Iter ", i
		for d in range(len(use)):
			for n in range(len(nodes)):
				fr = nodes[n].friends
				nexp[d, n] = sum(map(lambda x: expect[d, x] * pmap[d, n, x], fr)) + 1
			# if shps[d,n] > expect[d,n]:
			# 	print shps[d,n], expect[d,n], d, n
			# 	return
		expect = np.copy(nexp)
		nexp = np.zeros(expect.shape)
	return expect


def get_expectations(nodes, shps=None, pmap=None, iters=10):
	"""
	Creates Expectation Numpy Array: [destination, start]
	:param nodes: Full Node array
	:param nadj: Adjacency List speedup
	:param shps: Shortest Paths speedup
	:param pmap: Probability Map speedup
	:param iters: Convergence Iterations
	:return:
	"""
	if shps is None:
		shps = shortest_paths(nodes)
	if pmap is None:
		pmap = make_pmap(nodes)
	expect = np.copy(shps)
	nexp = np.zeros(expect.shape)
	for i in range(iters):
		print "Starting Iter ", i
		for d in nodes:
			for n in range(len(nodes)):
				fr = nodes[n].friends
				nexp[d, n] = sum(map(lambda x: expect[d, x] * pmap[d, n, x], fr)) + 1
			# if shps[d,n] > expect[d,n]:
			# 	print shps[d,n], expect[d,n], d, n
			# 	return
		expect = np.copy(nexp)
		nexp = np.zeros(expect.shape)
	return expect


def extract_instances_full(nodes):
	instances = []
	for i in range(len(nodes)):
		sour = int(random.random() * len(nodes))
		s = nodes[sour]
		dest = int(random.random() * len(nodes))
		d = nodes[dest]
		for f in s.friends:
			instances.append({"source": s, "destination": d, "step": nodes[f]})
	return instances


def extract_instances_random(nodes, num):
	instances = []
	for i in range(num):
		sour = int(random.random() * len(nodes))
		s = nodes[sour]
		dest = int(random.random() * len(nodes))
		d = nodes[dest]
		step = int(random.random() * len(s.friends))
		f = nodes[s.friends[step]]
		instances.append({"source": s, "destination": d, "step": f})
	return instances


def extract_feature(instance):
	destination = instance["destination"]
	step = instance["step"]
	dist = 1. / (distance(step.pos, destination.pos) + 1)

	communities_in_common = len(set.intersection(set(step.comm), set(destination.comm)))

	degree = step.deg1

	a = .001
	t = 275
	dist_weighted = 1. / (math.tanh(a * (dist - t)) + 1)
	log_degree = math.log(degree)

	# DISTANCE should be 1/(dist+epsilon)
	# Weighted Dist should be 1/(wd+epsilon)

	return np.array([dist, degree, communities_in_common, dist_weighted, log_degree])


def extract_feature_matrix(instances):
	feature_transform = []
	for i in instances:
		source = i["source"]
		destination = i["destination"]
		step = i["step"]
		dist = 1. / (distance(step.pos, destination.pos) + 1)

		communities_in_common = len(set.intersection(set(step.comm), set(destination.comm)))

		degree = step.deg1

		a = .001
		t = 275
		dist_weighted = 1. / (math.tanh(a * (dist - t)) + 1)
		log_degree = math.log(degree)

		# DISTANCE should be 1/(dist+epsilon)
		# Weighted Dist should be 1/(wd+epsilon)

		feature_transform.append([dist, degree, communities_in_common, dist_weighted, log_degree])
	return np.array(feature_transform)



def logistic_regression(fms, ys):
	l = LogisticRegression()
	print fms
	l.fit(fms, ys)
	weights = l.coef_[0]
	return weights


"""
5000
>>> ss.ks_2samp(str_loc.flatten(),str_unif.flatten())
Ks_2sampResult(statistic=0.11050000000000004, pvalue=1.2305905054301375e-53)
>>> ss.ks_2samp(str_weighted.flatten(),str_unif.flatten())
Ks_2sampResult(statistic=0.16189999999999999, pvalue=1.1850542278368712e-114)
>>> ss.ks_2samp(str_weighted.flatten(),str_loc.flatten())
Ks_2sampResult(statistic=0.12550000000000003, pvalue=4.6045356614998164e-69)
>>> str_weighted.sum()/(100**2)
3.4120704641392616
>>> str_loc.sum()/(100**2)
3.4489592996780862
>>> str_unif.sum()/(100**2)
3.5254522545667095
>>> ss.entropy(str_weighted.flatten(),str_unif.flatten())
0.00018068838354851744
>>> ss.entropy(str_weighted.flatten(),str_loc.flatten())
0.00089896771895634821
>>> ss.entropy(str_unif.flatten(),str_loc.flatten())
0.00049223800285286886

1000
>>> ss.ks_2samp(str_loc.flatten(),str_unif.flatten())
Ks_2sampResult(statistic=0.023112999999999939, pvalue=1.6496318300849063e-232)
>>> ss.ks_2samp(str_loc.flatten(),str_weighted.flatten())
Ks_2sampResult(statistic=0.12986999999999999, pvalue=0.0)
>>> ss.ks_2samp(str_unif.flatten(),str_weighted.flatten())
Ks_2sampResult(statistic=0.14990400000000004, pvalue=0.0)
>>> str_weighted.sum()/1000**2
2.8645705752528485
>>> str_loc.sum()/1000**2
2.9445184069483004
>>> str_unif.sum()/1000**2
2.9634520830754081
>>> ss.entropy(str_unif.flatten(),str_loc.flatten())
0.00023296184311607788
>>> ss.entropy(str_weighted.flatten(),str_loc.flatten())
0.00044082082163467222
>>> ss.entropy(str_weighted.flatten(),str_unif.flatten())
0.00011261765378879981

"""
