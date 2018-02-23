import numpy as np
import time
import cProfile
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA


def cross_entropy_error_from_file(w, data, labels):
	se = 0
	sg = 0
	sh = 0
	count=0
	with open(data) as f:
		with open(labels) as g:
			x=f.readline()
			y=g.readline()
			while x:
				x=np.fromstring(x,dtype=float,sep=' ')
				y=float(y)
				exppart=np.e**(y*np.dot(w,x))
				part2 = -y/(1.+exppart)
				se += np.log(1. + 1./exppart)
				sg += x*part2
				sh += -part2**2*np.outer(x,x)*exppart
				x=f.readline()
				y=g.readline()
				count+=1
	return 1./float(count) * se, 1./float(count) * sg, 1./float(count) * sh

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


def lm_opt_onepass(w, f, data, y, conv):
	eta = 1.
	fn=10000000.
	while fn>conv:
		t=time.time()
		fp = np.copy(fn)
		fn, grad, hess = f(w,data,y)
		if fp < fn:
			eta *= 10.
		else:
			eta *= .1
		w += np.dot(np.linalg.pinv(hess - eta*np.identity(w.shape[0])),grad)
		print "Iter comlplete with ", eta, time.time()-t
	return w

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
		if np.dot(self.w, x)>=0:
			return 1.
		return -1.

	def final(self):
		self.w = np.dot(np.linalg.pinv(self.xtx),self.xty)
		return self.w

	def regressFromFile(self,data,labels):
		with open(data) as f:
			with open(labels) as g:
				x=f.readline()
				y=g.readline()
				while x:
					self.addPoint(np.fromstring(x, dtype=float, sep=' '), float(y))
					x=f.readline()
					y=g.readline()
		return self.final()

class ExternalLogisticRegressor:

	def __init__(self):
		self.init_reg = ExternalRegressor()
		self.w = 0

	def classify(self,x):
		if np.dot(self.w, x)>=0:
			return 1.
		return -1.

	def regressFromFile(self,data,labels):
		t=time.time()
		ws = self.init_reg.regressFromFile(data, labels)
		print "Lin Reg complete", time.time()-t
		self.w = lm_opt_onepass(ws,cross_entropy_error_from_file,data,labels, 0.001)
		return self.w


def calculate_class_error(Xs,ys,cfunc):
	hs = map(cfunc, Xs)
	err_num=sum(((ys-hs)/2.)**2)
	return float(err_num)/float(Xs.shape[0])


def easy_lin_regressor_unit_test():
	er = ExternalLogisticRegressor()

	ws= er.regressFromFile("../../rand_xs.txt", "../../rand_ys.txt")

	# Xs = np.loadtxt("rand_xs.txt")
	# ys = np.loadtxt("rand_ys.txt")
	#
	# pca = PCA(11)
	#
	# Zs = pca.fit_transform(Xs)
	#
	# print ws
	#
	# plt.scatter(Zs[:,0], Zs[:,1], c=ys)
	#
	# zws = pca.transform([ws])
	#
	# delta = 0.025
	# xsp = np.arange(-1.0, 1.0, delta)
	# ysp = np.arange(-1.0, 1.0, delta)
	# Xsp, Ysp = np.meshgrid(xsp, ysp)
	# Zsp = zws[0,0]*Xsp + zws[0,1]*Ysp
	#
	# plt.contour(Xsp, Ysp, Zsp, [0])
	#
	# plt.show()
	#print calculate_class_error(Xs, ys, lambda x: er.classify(x))

cProfile.run("easy_lin_regressor_unit_test()")