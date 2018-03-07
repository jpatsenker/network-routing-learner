import numpy as np
import time
import cProfile
from multiprocessing import Process
from multiprocessing import Manager

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_split_points(file,num_pts):
	splits=[0]
	c=0
	i=0
	with open(file) as f:
		line = f.readline()
		while line:
			c+=len(line)
			i+=1
			if i==num_pts:
				splits.append(c)
				i=0
			line = f.readline()
	return splits

def delegate_cross_entropy_error_from_file(w, f, lim, ret, pnum):
	se=0
	sg=0
	sh=0
	count=0
	line=f.readline()
	while count<lim and line:
		count+=1.
		line=np.fromstring(line,dtype=float,sep=' ')
		x=line[:-1]
		y=line[-1]
		exppart=np.e**(y*np.dot(w,x))
		part2 = -y/(1.+exppart)
		invc=1./count
		se += (np.log(1. + 1./exppart)-se)*invc
		sg += (x*part2-sg)*invc
		sh += (-part2**2*np.outer(x,x)*exppart-sh)*invc
		line=f.readline()
	ret[pnum]=[se, sg, sh]

def cross_entropy_error_from_file_multithreaded(w, data, splits,pnum):
	se = [0]*(len(splits)-1)
	sg = [0]*(len(splits)-1)
	sh = [0]*(len(splits)-1)
	m = Manager()
	ret = m.list([[0,0,0]]*(len(splits)-1))
	ps = []
	fs = [open(data, "r") for i in range(len(splits)-1)]
	cores=len(splits)-1
	for i in range(1,len(splits)):
		fs[i-1].seek(splits[i-1])
		p=Process(target=delegate_cross_entropy_error_from_file, args=(w, fs[i-1], pnum, ret,i-1))
		ps.append(p)
		p.start()

	for i in range(len(ps)):
		ps[i].join()
		se[i], sg[i], sh[i] = ret[i]

	return 1./cores * sum(se), 1./cores * sum(sg), 1./cores * sum(sh)

def cross_entropy_error_from_file(w, data):
	se = 0
	sg = 0
	sh = 0
	count=0.
	with open(data) as f:
		line=f.readline()
		while line:
			count+=1.
			line=np.fromstring(line,dtype=float,sep=' ')
			x=line[:-1]
			y=line[-1]
			exppart=np.e**(y*np.dot(w,x))
			part2 = -y/(1.+exppart)
			invc=1./count
			se += (np.log(1. + 1./exppart)-se)*invc
			sg += (x*part2-sg)*invc
			sh += (-part2**2*np.outer(x,x)*exppart-sh)*invc
			x=f.readline()
	return se, sg, sh

# def grad_cross_entropy_error_from_file(w, data, labels):
# 	ss = np.zeros(w.shape)
# 	count=0
# 	with open(data) as f:
# 		with open(labels) as g:
# 			x=f.readline()
# 			y=g.readline()
# 			while x:
# 				x=np.fromstring(x,dtype=float,sep=' ')
# 				y=float(y)
# 				ss += (-y*x)/(1.+np.e**(y*np.dot(w,x)))
# 				x=f.readline()
# 				y=g.readline()
# 				count+=1
# 	return 1/float(count) * ss
#
# def hess_cross_entropy_error_from_file(w, data, labels):
# 	ss = np.zeros([w.shape[0],w.shape[0]])
# 	count=0
# 	with open(data) as f:
# 		with open(labels) as g:
# 			x=f.readline()
# 			y=g.readline()
# 			while x:
# 				x=np.fromstring(x,dtype=float,sep=' ')
# 				y=float(y)
# 				ss += -(y)**2*np.outer(x,x)*np.e**(y*np.dot(w,x))/((1.+np.e**(y*np.dot(w,x)))**2)
# 				x=f.readline()
# 				y=g.readline()
# 				count+=1
# 	return 1/float(count) * ss

def cross_entropy_error(w, x, y):
	ss = 0
	i = 0
	while i < x.shape[0]:
		ss += np.log(1. + np.e**(-y[i]*np.dot(w,x[i])))
		i += 1
	return 1./float(x.shape[0]) * ss


def grad_cross_entropy_error(w, x, y):
	ss = np.zeros(w.shape)
	i = 0
	while i<x.shape[0]:
		ss += (-y[i]*x[i])/(1.+np.e**(y[i]*np.dot(w,x[i])))
		i += 1
	return 1/float(x.shape[0]) * ss

def hess_cross_entropy_error(w, x, y):
	ss = np.zeros([w.shape[0],w.shape[0]])
	i = 0
	while i<x.shape[0]:
		ss += -(y[i])**2*np.outer(x[i],x[i])*np.e**(y[i]*np.dot(w,x[i]))/((1.+np.e**(y[i]*np.dot(w,x[i])))**2)
		i += 1
	return 1/float(x.shape[0]) * ss



def l2norm(x):
	return np.sqrt(np.sum(x**2))

def lm_update(w, grad, hess, p, eta, y):
	iden = np.identity(w.shape[0])
	wn = w + np.dot(np.linalg.pinv(hess(w,p,y) - eta*iden),grad(w,p,y))
	return wn

def lm_opt(w, f, grad, hess, data, y,conv):
	t=time.time()
	eta = 1.
	fn=10000000.
	while fn>conv:
		fp = np.copy(fn)
		w = lm_update(w, grad, hess, data, eta, y)
		fn = f(w,data,y)
		if fp < fn:
			eta *= 10.
		else:
			eta *= .1
		print "Iter comlplete with ", eta, time.time()-t
	return w


def lm_opt_onepass(w, f, data, conv):
	eta = 1.
	fn=10000000.
	fp=0.
	while l2norm(fn-fp)>conv:
		t=time.time()
		fp = np.copy(fn)
		fn, grad, hess = f(w,data)
		if fp < fn:
			eta *= 10.
		else:
			eta *= .1
		w += np.dot(np.linalg.pinv(hess - eta*np.identity(w.shape[0])),grad)
		print "Iter complete with ", eta, time.time()-t
	return w

def delegateRegress(f,numpoints,ret,pnum):
	print numpoints
	line=f.readline()
	xtx=0.
	xty=0.
	c=0
	while line and c<numpoints:
		d=np.fromstring(line, dtype=float, sep=' ')
		x=d[:-1]
		y=d[-1]
		xtx+=np.outer(x,x)
		xty+=int(y)*x
		f.readline()
		c+=1
#		if c%100==0:
#			print c
	print "p done", pnum
	ret[pnum]=[xtx,xty]


class ExternalRegressor:

	def __init__(self):
		self.xtx = 0
		self.xty = 0
		self.w = 0

	def addPoint(self, x, y):
		"""
		adds point to the regressor
		:param x: list of features
		:param y: label
		"""
		self.xtx += np.outer(x,x)
		self.xty += int(y)*x

	def classify(self,x):
		if np.dot(self.w, x) >= 0:
			return 1.
		return -1.

	def final(self):
		self.w = np.dot(np.linalg.pinv(self.xtx),self.xty)
		return self.w

	def regressFromFile(self,data):
		with open(data) as f:
			line=f.readline()
			while line:
				x=np.fromstring(line, dtype=float, sep=' ')
				self.addPoint(x[:-1],x[-1])
				line=f.readline()
		return self.final()

	def regressFromFileMultithreaded(self,data,splits,num_points):
		xtx = [0]*(len(splits)-1)
		xty = [0]*(len(splits)-1)
		m = Manager()
		ret = m.list([[0,0]]*(len(splits)-1))
		ps = []
		fs = [open(data, "r") for i in range(len(splits)-1)]
		cores=len(splits)-1
		for i in range(1,len(splits)):
			fs[i-1].seek(splits[i-1])
			p=Process(target=delegateRegress, args=(fs[i-1], num_points, ret,i-1))
			ps.append(p)
			p.start()
			print "start", p
		for i in range(len(ps)):
			ps[i].join()
			xtx[i], xty[i] = ret[i]
		self.xtx=np.sum(xtx,axis=0)
		self.xty=np.sum(xty,axis=0)
		return self.final()

class ExternalLogisticRegressor:

	def __init__(self):
		self.init_reg = ExternalRegressor()
		self.w = 0

	def classify(self,x):
		if np.dot(self.w, x)>=0:
			return 1.
		return -1.

	def regressFromFile(self,data):
		t=time.time()
		ws = self.init_reg.regressFromFile(data)
		print "Lin Reg complete", time.time()-t
		self.w = lm_opt_onepass(ws,cross_entropy_error_from_file,data, 0.1)
		return self.w

	def regressFromFileMultithreaded(self,data):
		t=time.time()
		NPTS=4622000
		NDIV=2
		splits = get_split_points(data, NPTS/NDIV)
		print "splits calc: ", time.time()-t
		t=time.time()
		ws = self.init_reg.regressFromFileMultithreaded(data,splits,NPTS/NDIV)
		print "Lin Reg complete", time.time()-t
		self.w = lm_opt_onepass(ws,(lambda wi, datai: cross_entropy_error_from_file_multithreaded(wi, datai, splits,NPTS/NDIV)),data, 0.01)
		return self.w


def calculate_class_error(Xs,ys,cfunc):
	hs = map(cfunc, Xs)
	err_num=sum(((ys-hs)/2.)**2)
	return float(err_num)/float(Xs.shape[0])

# def calculate_class_error_from_file(f,g,cfunc):
# 	with open(data) as f:
# 		with open(labels) as g:
# 			x=f.readline()
# 			y=g.readline()
# 			errnum=0
# 			i=0
# 			while x:
# 				i+=1
# 				errnum=((y-cfunc(x))/2.)**2
# 				x=f.readline()
# 				y=g.readline()
# 	return errnum

def long_lin_regressor_unit_test():
	er = ExternalLogisticRegressor()

	ws = er.regressFromFileMultithreaded("big_dat.txt")

def run():
	er = ExternalLogisticRegressor()

	ws = er.regressFromFileMultithreaded("gowalla_ml_dataset.txt")


def easy_lin_regressor_unit_test():
	er = ExternalLogisticRegressor()

	ws = er.regressFromFileMultithreaded("rand_data_head.txt")
	#ws= er.regressFromFile("../../rand_xs.txt", "../../rand_ys.txt")

	dat = np.loadtxt("rand_data_head.txt")
	Xs = dat[:,:-1]
	ys = dat[:,-1]

	pca = PCA(11)

	Zs = pca.fit_transform(Xs)

	print ws

	plt.scatter(Zs[:,0], Zs[:,1], c=ys)

	zws = pca.transform([ws])

	delta = 0.025
	xsp = np.arange(-1.0, 1.0, delta)
	ysp = np.arange(-1.0, 1.0, delta)
	Xsp, Ysp = np.meshgrid(xsp, ysp)
	Zsp = zws[0,0]*Xsp + zws[0,1]*Ysp

	plt.contour(Xsp, Ysp, Zsp, [0])

	plt.show()
	#print calculate_class_error_from_file(Xs, ys, lambda x: er.classify(x))


#cProfile.run("easy_lin_regressor_unit_test()")
cProfile.run("run()")