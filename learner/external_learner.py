import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def cross_entropy_error(w, x, y):
	ss = 0
	i = 0
	while i < x.shape[0]:
		ss += np.log(1. + math.e**(-y[i]*np.dot(w,x[i])))
		i += 1
	return 1./float(x.shape[0]) * ss


def grad_cross_entropy_error(w, x, y):
	ss = np.zeros(w.shape)
	i = 0
	while i<x.shape[0]:
		ss += (-y[i]*x[i])/(1.+math.e**(y[i]*np.dot(w,x[i])))
		i += 1
	#print ss
	return 1/float(x.shape[0]) * ss

def hess_cross_entropy_error(w, x, y):
	ss = np.zeros([w.shape[0],w.shape[0]])
	i = 0
	while i<x.shape[0]:
		print ((1.+np.e**(y[i]*np.dot(w,x[i])))**2)
		ss += -(y[i])**2*np.outer(x[i],x[i])*np.e**(y[i]*np.dot(w,x[i]))/((1.+np.e**(y[i]*np.dot(w,x[i])))**2)
		i += 1
	return 1/float(x.shape[0]) * ss

def l2norm(x):
	return np.sqrt(np.sum(x**2))

def lm_update(w, grad, hess, p, eta, y):
	iden = np.identity(w.shape[0])
	#print hess(w,p,y)
	wn = w + np.dot(np.linalg.pinv(hess(w,p,y) - eta*iden),grad(w,p,y))
	return wn

def lm_opt(w, f, grad, hess, data, y,conv):
	eta = 1.
	error = f(w,data,y)
	w_prev = np.ones(w.shape) * 10000000.
	fn=10000000.
	while l2norm(w-w_prev)>conv:
		print "iter with ", eta, l2norm(w-w_prev), conv
		w_prev = np.copy(w)
		fp = np.copy(fn)
		w = lm_update(w, grad, hess, data, eta, y)
		fn = f(w,data,y)
		print "error:", fn, fp
		if fp < fn:
			eta *= 10.
		else:
			eta *= .1
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
		ws = self.init_reg.regressFromFile(data, labels)
		Xs = np.loadtxt(data)
		ys = np.loadtxt(labels)
		self.w = lm_opt(ws,cross_entropy_error,grad_cross_entropy_error,hess_cross_entropy_error,Xs,ys, .01)
		return self.w


def calculate_class_error(Xs,ys,cfunc):
	hs = map(cfunc, Xs)
	err_num=sum(((ys-hs)/2.)**2)
	return float(err_num)/float(Xs.shape[0])


def easy_lin_regressor_unit_test():
	er = ExternalLogisticRegressor()

	ws= er.regressFromFile("rand_xs.txt", "rand_ys.txt")

	Xs = np.loadtxt("rand_xs.txt")
	ys = np.loadtxt("rand_ys.txt")

	pca = PCA(11)

	Zs = pca.fit_transform(Xs)

	print ws

	plt.scatter(Zs[:,0], Zs[:,1], c=ys)

	zws = pca.transform([ws])

	print zws
	delta = 0.025
	xsp = np.arange(-1.0, 1.0, delta)
	ysp = np.arange(-1.0, 1.0, delta)
	Xsp, Ysp = np.meshgrid(xsp, ysp)
	Zsp = zws[0,0]*Xsp + zws[0,1]*Ysp

	plt.contour(Xsp, Ysp, Zsp, [0])

	plt.show()
	print calculate_class_error(Xs, ys, lambda x: er.classify(x))

easy_lin_regressor_unit_test()