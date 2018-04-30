import numpy as np
import time
import sys
import os
import cProfile
from multiprocessing import Process
from multiprocessing import Manager

#import matplotlib.pyplot as plt
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


def get_split_norm_points(file,num_pts):
	splits=[0]
	top = 0
	bottom = 0
	c=0
	i=0
	with open(file) as f:
		line = f.readline()
		while line:
			c+=len(line)
			i+=1
			line_np =np.fromstring(line, dtype=float, sep=' ')
			top = np.maximum(top,line_np[:-2])
			bottom = np.minimum(bottom, line_np[:-2])
			if i==num_pts:
				splits.append(c)
				print i, time.time()
				sys.stdout.flush()
				i=0
			line = f.readline()
	return splits, top, bottom

def get_norm_points(file):
	splits=[0]
	top = 0
	bottom = 0
	for i in range(50):
		with open(file + str(i) + ".txt") as f:
			line = f.readline()
			c=0
			while line:
				line_np =np.fromstring(line, dtype=float, sep=' ')
				top = np.maximum(top,line_np[:-2])
				bottom = np.minimum(bottom, line_np[:-2])
				line = f.readline()
				c+=1
				if c%1000000==0:
					print c
					sys.stdout.flush()
		print "next"
		sys.stdout.flush()
	np.savetxt("top.txt", top)
	np.savetxt("bottom.txt", bottom)



def delegate_cross_entropy_error_from_file(w1, w2, f, lim, ret, pnum,top,bottom):
	os.system("taskset -p -c " + str(pnum) + " " + str(os.getpid()))
	se1=0
	sg1=0
	sh1=0
	se2=0
	sg2=0
	sh2=0
	count=0
	line=f.readline()
	while count<lim and line:
		count+=1.
		line=np.fromstring(line,dtype=float,sep=' ')
		x=(line[:-2]-bottom)/top
		y1=(line[-2]-0.5)*2.
		y2=(line[-1]-0.5)*2.
		exppart_1=np.e**(y1*np.dot(w1,x))
		exppart_2=np.e**(y2*np.dot(w2,x))
		part2_1 = -y1/(1.+exppart_1)
		part2_2 = -y2/(1.+exppart_2)
		invc=1./count
		se1 += (np.log(1. + 1./exppart_1)-se1)*invc
		sg1 += (x*part2_1-sg1)*invc
		sh1 += (-part2_1**2*np.outer(x,x)*exppart_1-sh1)*invc
		se2 += (np.log(1. + 1./exppart_2)-se2)*invc
		sg2 += (x*part2_2-sg2)*invc
		sh2 += (-part2_2**2*np.outer(x,x)*exppart_2-sh2)*invc
		line=f.readline()
	ret[pnum]=[se1, sg1, sh1, se2, sg2, sh2]

def delegate_cross_entropy_error_from_multiple_files(w1, w2, f, ret, pnum,top,bottom):
	os.system("taskset -p -c " + str(pnum) + " " + str(os.getpid()))
	se1=0
	sg1=0
	sh1=0
	se2=0
	sg2=0
	sh2=0
	count=0
	line=f.readline()
	while line:
		if count%100000==0:
			print count
			sys.stdout.flush()
		count+=1.
		line=np.fromstring(line,dtype=float,sep=' ')
		x=(line[:-2]-bottom)/top
		y1=(line[-2]-0.5)*2.
		y2=(line[-1]-0.5)*2.
		exppart_1=np.e**(y1*np.dot(w1,x))
		exppart_2=np.e**(y2*np.dot(w2,x))
		part2_1 = -y1/(1.+exppart_1)
		part2_2 = -y2/(1.+exppart_2)
		invc=1./count
		se1 += (np.log(1. + 1./exppart_1)-se1)*invc
		sg1 += (x*part2_1-sg1)*invc
		sh1 += (-part2_1**2*np.outer(x,x)*exppart_1-sh1)*invc
		se2 += (np.log(1. + 1./exppart_2)-se2)*invc
		sg2 += (x*part2_2-sg2)*invc
		sh2 += (-part2_2**2*np.outer(x,x)*exppart_2-sh2)*invc
		line=f.readline()
	ret[pnum]=[se1, sg1, sh1, se2, sg2, sh2]

def delegate_cross_entropy_error_from_multiple_files_bins(w1, w2, f, ret, pnum,top,bottom,bins=[0,50,100,200,400,800,1600,3200,6400,12800,20000]):
	os.system("taskset -p -c " + str(pnum) + " " + str(os.getpid()))
	se1 = np.zeros([len(bins)-1])
	sg1 = np.zeros([len(bins)-1,len(top)])
	sh1 = np.zeros([len(bins)-1,len(top),len(top)])
	se2 = np.zeros([len(bins)-1])
	sg2 = np.zeros([len(bins)-1,len(top)])
	sh2 = np.zeros([len(bins)-1,len(top),len(top)])
	count=0
	line=f.readline()
	while line:
		if count%100000==0:
			print count
			sys.stdout.flush()
		count+=1.
		line=np.fromstring(line,dtype=float,sep=' ')
		x=(line[:-2]-bottom)/top
		y1=(line[-2]-0.5)*2.
		y2=(line[-1]-0.5)*2.
		for b in range(len(bins)-1):
			if x[0] > bins[b] and x[1]<bins[b+1]:
				exppart_1=np.e**(y1*np.dot(w1[b],x))
				exppart_2=np.e**(y2*np.dot(w2[b],x))
				part2_1 = -y1/(1.+exppart_1)
				part2_2 = -y2/(1.+exppart_2)
				invc=1./count
				se1[b] += (np.log(1. + 1./exppart_1)-se1[b])*invc
				sg1[b] += (x*part2_1-sg1[b])*invc
				sh1[b] += (-part2_1**2*np.outer(x,x)*exppart_1-sh1[b])*invc
				se2[b] += (np.log(1. + 1./exppart_2)-se2[b])*invc
				sg2[b] += (x*part2_2-sg2[b])*invc
				sh2[b] += (-part2_2**2*np.outer(x,x)*exppart_2-sh2[b])*invc
				continue
		line=f.readline()
	for b in range(len(bins)-1):
		ret[b][pnum]=[se1[b], sg1[b], sh1[b], se2[b], sg2[b], sh2[b]]

def cross_entropy_error_from_file_multithreaded(w1, w2, data, splits,pnum,top,bottom):
	se1 = [0]*(len(splits)-1)
	sg1 = [0]*(len(splits)-1)
	sh1 = [0]*(len(splits)-1)
	se2 = [0]*(len(splits)-1)
	sg2 = [0]*(len(splits)-1)
	sh2 = [0]*(len(splits)-1)
	m = Manager()
	ret = m.list([[0,0,0,0,0,0]]*(len(splits)-1))
	ps = []
	fs = [open(data, "r") for i in range(len(splits)-1)]
	cores=len(splits)-1
	for i in range(1,len(splits)):
		fs[i-1].seek(splits[i-1])
		p=Process(target=delegate_cross_entropy_error_from_file, args=(w1, w2, fs[i-1], pnum, ret,i-1,top,bottom))
		ps.append(p)
		p.start()

	for i in range(len(ps)):
		ps[i].join()
		se1[i], sg1[i], sh1[i], se2[i], sg2[i], sh2[i] = ret[i]

	return 1./cores * sum(se1), 1./cores * sum(sg1), 1./cores * sum(sh1), 1./cores * sum(se2), 1./cores * sum(sg2), 1./cores * sum(sh2)

def cross_entropy_error_from_multifile_multithreaded(w1, w2, data,top,bottom):
	se1 = [0]*50
	sg1 = [0]*50
	sh1 = [0]*50
	se2 = [0]*50
	sg2 = [0]*50
	sh2 = [0]*50
	m = Manager()
	ret = m.list([[0,0,0,0,0,0]]*50)
	ps = []
	fs = [open(data + str(i) + ".txt", "r") for i in range(50)]
	cores=50
	for i in range(50):
		p=Process(target=delegate_cross_entropy_error_from_multiple_files, args=(w1, w2, fs[i], ret, i,top,bottom))
		ps.append(p)
		p.start()

	for i in range(len(ps)):
		ps[i].join()
		se1[i], sg1[i], sh1[i], se2[i], sg2[i], sh2[i] = ret[i]

	return 1./cores * sum(se1), 1./cores * sum(sg1), 1./cores * sum(sh1), 1./cores * sum(se2), 1./cores * sum(sg2), 1./cores * sum(sh2)

def cross_entropy_error_from_multifile_multithreaded_bins(w1, w2, data,top,bottom,bins=[0,50,100,200,400,800,1600,3200,6400,12800,20000]):
	se1 = np.zeros([len(bins)-1,50])
	sg1 = np.zeros([len(bins)-1,50,len(top)])
	sh1 = np.zeros([len(bins)-1,50,len(top),len(top)])
	se2 = np.zeros([len(bins)-1,50])
	sg2 = np.zeros([len(bins)-1,50,len(top)])
	sh2 = np.zeros([len(bins)-1,50,len(top),len(top)])
	m = Manager()
	ret = m.list([[[0,0,0,0,0,0]]*50]*(len(bins)-1))
	ps = []
	fs = [open(data + str(i) + ".txt", "r") for i in range(50)]
	cores=50.
	for i in range(50):
		p=Process(target=delegate_cross_entropy_error_from_multiple_files_bins, args=(w1, w2, fs[i], ret, i,top,bottom,bins))
		ps.append(p)
		p.start()

	for i in range(len(ps)):
		ps[i].join()
		for b in range(len(bins)-1):
			se1[b,i], sg1[b,i], sh1[b,i], se2[b,i], sg2[b,i], sh2[b,i] = ret[b][i]

	return 1./cores * np.sum(se1,axis=1), 1./cores * np.sum(sg1,axis=1), 1./cores * np.sum(sh1,axis=1), 1./cores * np.sum(se2,axis=1), 1./cores * np.sum(sg2,axis=1), 1./cores * np.sum(sh2,axis=1)

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
			line=f.readline()
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

def lm_opt_onepass_2targ(w1, w2, f, data, conv, fil="temp_dump_gowalla.txt"):
	eta1 = 1.
	eta2 = 1.
	fn1=10000000.
	fp1=0.
	fn2=10000000.
	fp2=0.
	c=0
	while l2norm(fn1-fp1)>conv and l2norm(fn2-fp2)>conv:
		t=time.time()
		fp1 = np.copy(fn1)
		fp2 = np.copy(fn2)
		fn1, grad1, hess1, fn2, grad2, hess2 = f(w1,w2,data)
		if fp1 < fn1:
			eta1 *= 10.
		else:
			eta1 *= .1
		if fp2 < fn2:
			eta2 *= 10.
		else:
			eta2 *= .1
		#print hess1.shape, grad1.shape, eta1, w1.shape
		w1 += np.dot(np.linalg.pinv(hess1 - eta1*np.identity(w1.shape[0])),grad1)
		w2 += np.dot(np.linalg.pinv(hess2 - eta2*np.identity(w2.shape[0])),grad2)
		c+=1
		with open(fil, 'a') as wrtr:
			wrtr.write("Iteration " + str(c) + " Complete with " + str(eta1) + " " + str(eta2) + "\n")
			wrtr.write(str(w1) + "\n" + str(w2) + "\n")
		print "Iter complete with ", eta1, eta2, time.time()-t
	return w1, w2

def lm_opt_onepass_2targ_bins(w1, w2, f, data, conv, fil="temp_dump_airnet_bins.txt", bins = [0,50,100,200,400,800,1600,3200,6400,12800,20000]):
	eta1 = [1.]*(len(bins)-1)
	eta2 = [1.]*(len(bins)-1)
	fn1=[10000000.]*(len(bins)-1)
	fp1=[0.]*(len(bins)-1)
	fn2=[10000000.]*(len(bins)-1)
	fp2=[0.]*(len(bins) - 1)
	c=0
	while l2norm(fn1[0]-fp1[0])>conv and l2norm(fn2[0]-fp2[0])>conv:
		t=time.time()
		fp1 = np.copy(fn1)
		fp2 = np.copy(fn2)
		fn1, grad1, hess1, fn2, grad2, hess2 = f(w1,w2,data)
		for b in range(len(bins)-1):
			if fp1[b] < fn1[b]:
				eta1[b] *= 10.
			else:
				eta1[b] *= .1
			if fp2[b] < fn2[b]:
				eta2[b] *= 10.
			else:
				eta2[b] *= .1
			#print hess1.shape, grad1.shape, eta1, w1.shape
			w1[b] += np.dot(np.linalg.pinv(hess1[b] - eta1[b]*np.identity(w1[b].shape[0])),grad1[b])
			w2[b] += np.dot(np.linalg.pinv(hess2[b] - eta2[b]*np.identity(w2[b].shape[0])),grad2[b])
		c+=1
		with open(fil, 'a') as wrtr:
			wrtr.write("Iteration " + str(c) + " Complete with " + str(eta1) + " " + str(eta2) + "\n")
			wrtr.write(str(w1) + "\n" + str(w2) + "\n")
		print "Iter complete with ", eta1, eta2, time.time()-t
	return w1, w2

def delegateRegress(f,numpoints,ret,pnum,top,bottom):
	print numpoints
	line=f.readline()
	xtx=0.
	xty1=0.
	xty2=0.
	c=0
	while line and c<numpoints:
		d=np.fromstring(line, dtype=float, sep=' ')
		x=(d[:-2]-bottom)/top
		y1=2.*(d[-2]-0.5)
		y2=2.*(d[-1]-0.5)
		xtx+=np.outer(x,x)
		xty1+=int(y1)*x
		xty2+=int(y2)*x
		f.readline()
		c+=1
#		if c%100==0:
#			print c
	print "p done", pnum
	print xtx, xty1, xty2
	ret[pnum]=[xtx,xty1,xty2]

def delegateRegressFullFile(f,ret,pnum,top,bottom):
	os.system("taskset -p -c " + str(pnum) + " " + str(os.getpid()))
	line=f.readline()
	xtx=0.
	xty1=0.
	xty2=0.
	c=0
	while line:
		if c%1000==100000:
			print c, time.time()
			sys.stdout.flush()
		sys.stdout.flush()
		d=np.fromstring(line, dtype=float, sep=' ')
		x=(d[:-2]-bottom)/top
		y1=2.*(d[-2]-0.5)
		y2=2.*(d[-1]-0.5)
		xtx+=np.outer(x,x)
		xty1+=int(y1)*x
		xty2+=int(y2)*x
		line = f.readline()
		c+=1

	print "p done", pnum
	ret[pnum]=[xtx,xty1,xty2]

def delegateRegressFullFileBins(f,ret,pnum,top,bottom,bins=[0,50,100,200,400,800,1600,3200,6400,12800,20000]):
	os.system("taskset -p -c " + str(pnum) + " " + str(os.getpid()))
	line=f.readline()
	xtx=np.zeros([len(bins)-1,len(top),len(top)])
	xty1=np.zeros([len(bins)-1,len(top)])
	xty2=np.zeros([len(bins)-1,len(top)])
	c=0
	while line:
		if c%1000==100000:
			print c, time.time()
			sys.stdout.flush()
		sys.stdout.flush()
		d=np.fromstring(line, dtype=float, sep=' ')
		x=(d[:-2]-bottom)/top
		y1=2.*(d[-2]-0.5)
		y2=2.*(d[-1]-0.5)
		for b in range(len(bins)-1):
			if x[0] > bins[b] and x[1] < bins[b+1]:
				xtx[b]+=np.outer(x,x)
				xty1[b]+=int(y1)*x
				xty2[b]+=int(y2)*x
		line = f.readline()
		c+=1

	print "p done", pnum
	ret[pnum]=[xtx,xty1,xty2]


class ExternalRegressor:

	def __init__(self):
		self.xtx = 0
		self.xty1 = 0
		self.xty2 = 0
		self.w1 = 0
		self.w2 = 0

	def addPoint(self, x, y):
		"""
		adds point to the regressor
		:param x: list of features
		:param y: label
		"""
		self.xtx += np.outer(x,x)
		self.xty1 += int(y)*x
		self.xty2 += int(y)*x

	def classify(self,x):
		if np.dot(self.w, x) >= 0:
			return 1.
		return -1.

	def final(self):
		self.w1 = np.dot(np.linalg.pinv(self.xtx),self.xty1)
		self.w2 = np.dot(np.linalg.pinv(self.xtx),self.xty2)
		return self.w1, self.w2

	def regressFromFile(self,data):
		with open(data) as f:
			line=f.readline()
			while line:
				x=np.fromstring(line, dtype=float, sep=' ')
				self.addPoint(x[:-1],x[-1])
				line=f.readline()
		return self.final()

	def regressFromFileMultithreaded(self,data,splits,num_points,top,bottom):
		xtx = [0]*(len(splits)-1)
		xty1 = [0]*(len(splits)-1)
		xty2 = [0]*(len(splits)-1)
		m = Manager()
		ret = m.list([[0,0,0]]*(len(splits)-1))
		ps = []
		fs = [open(data, "r") for i in range(len(splits)-1)]
		cores=len(splits)-1
		for i in range(1,len(splits)):
			fs[i-1].seek(splits[i-1])
			p=Process(target=delegateRegress, args=(fs[i-1], num_points, ret,i-1,top,bottom))
			ps.append(p)
			p.start()
			print "start", p
		for i in range(len(ps)):
			ps[i].join()
			xtx[i], xty1[i], xty2[i] = ret[i]
		self.xtx=np.sum(xtx,axis=0)
		self.xty1=np.sum(xty1,axis=0)
		self.xty2=np.sum(xty2,axis=0)
		return self.final()

	def regressFromFileMultithreadedMultifile(self,data,top,bottom):
		xtx = [0]*50
		xty1 = [0]*50
		xty2 = [0]*50
		m = Manager()
		ret = m.list([[0,0,0]]*50)
		ps = []
		fs = [open(data + str(i) + ".txt", "r") for i in range(50)]
		cores=50
		for i in range(50):
			p=Process(target=delegateRegressFullFile, args=(fs[i], ret,i,top,bottom))
			ps.append(p)
			p.start()
			print "start", p
		for i in range(len(ps)):
			ps[i].join()
			xtx[i], xty1[i], xty2[i] = ret[i]
		self.xtx=np.sum(xtx,axis=0)
		self.xty1=np.sum(xty1,axis=0)
		self.xty2=np.sum(xty2,axis=0)
		return self.final()

	def regressFromFileMultithreadedMultifileBins(self,data,top,bottom,bins=[0,50,100,200,400,800,1600,3200,6400,12800,20000]):
		self.xtx=np.zeros([len(bins)-1,len(top),len(top)])
		self.xty1=np.zeros([len(bins)-1,len(top)])
		self.xty2=np.zeros([len(bins)-1,len(top)])
		xtx = np.zeros([50,len(bins)-1,len(top),len(top)])
		xty1 = np.zeros([50,len(bins)-1,len(top)])
		xty2 = np.zeros([50,len(bins)-1,len(top)])
		m = Manager()
		ret = m.list([[0,0,0]]*50)
		ps = []
		fs = [open(data + str(i) + ".txt", "r") for i in range(50)]
		cores=50
		for i in range(50):
			p=Process(target=delegateRegressFullFileBins, args=(fs[i], ret,i,top,bottom,bins))
			ps.append(p)
			p.start()
			print "start", p
		for i in range(len(ps)):
			ps[i].join()
			xtx[i], xty1[i], xty2[i] = ret[i]
		self.xtx=np.sum(xtx,axis=0)
		self.xty1=np.sum(xty1,axis=0)
		self.xty2=np.sum(xty2,axis=0)
		ws1 = np.zeros([len(bins)-1, len(top)])
		ws2 = np.zeros([len(bins)-1, len(top)])
		for i in range(len(bins)-1):
			ws2[i] = np.dot(np.linalg.pinv(self.xtx[i]),self.xty1[i])
			ws2[i] = np.dot(np.linalg.pinv(self.xtx[i]),self.xty2[i])
		return ws1, ws2

class ExternalLogisticRegressor:

	def __init__(self):
		self.init_reg = ExternalRegressor()
		self.w1 = 0
		self.w2 = 0

	def classify(self,x,w):
		if np.dot(w, x)>=0:
			return 1.
		return -1.

	def regressFromFile(self,data):
		t=time.time()
		ws = self.init_reg.regressFromFile(data)
		print "Lin Reg complete", time.time()-t
		self.w1,self.w2 = lm_opt_onepass(ws,cross_entropy_error_from_file,data, 0.1)
		return self.w1, self.w2

	def regressFromFileMultithreaded(self,data,fil="temp_dump.txt"):
		t=time.time()
		NPTS=527306
		NDIV=2
		splits,top,bottom = get_split_norm_points(data, NPTS/NDIV)
		#top,bottom = get_norms(data)
		print "splits calc: ", time.time()-t
		t=time.time()
		ws1,ws2 = self.init_reg.regressFromFileMultithreaded(data,splits,NPTS/NDIV,top,bottom)
		with open(fil, 'a') as wrtr:
			wrtr.write("Lin Regression Weights\n")
			wrtr.write(str(ws1) + "\n" + str(ws2) + "\n")
		print "Lin Reg complete", time.time()-t
		self.w1, self.w2 = lm_opt_onepass_2targ(ws1, ws2, (lambda wi1, wi2, datai: cross_entropy_error_from_file_multithreaded(wi1, wi2, datai, splits,NPTS/NDIV,top,bottom)),data, 0.01)
		return self.w1, self.w2

	def regressFromFileMultithreadedMultiFile(self,data,fil="temp_dump_airnet_bins.txt"):
		t=time.time()
		#top,bottom = get_norm_points(data)
		top,bottom=np.array([2.00092774e+04,   1.00000000e+03,   1.00000000e+00, 9.15000000e+02,   6.81892407e+00,   9.15000000e+02, 1.23771376e+07]), np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.])
		#top,bottom=np.array([8.439885362102702857e+03,9.999999999998809699e+02,1.000000000000000000e+00,8.185000000000000000e+03,9.010058489805235382e+00,8.185000000000000000e+03,5.089825838814330846e+07]),np.array([0.,0.,0.,0.,0.,0.,0.])
		print "splits calc: ", time.time()-t
		t=time.time()
		ws1,ws2 = self.init_reg.regressFromFileMultithreadedMultifile(data,top,bottom)
		with open(fil, 'a') as wrtr:
			wrtr.write("Lin Regression Weights\n")
			wrtr.write(str(ws1) + "\n" + str(ws2) + "\n")
		print "Lin Reg complete", time.time()-t
		self.w1, self.w2 = lm_opt_onepass_2targ(ws1, ws2, (lambda wi1, wi2, datai: cross_entropy_error_from_multifile_multithreaded(wi1, wi2, data,top,bottom)),data, 0.01)
		return self.w1, self.w2

	def regressFromFileMultithreadedMultiFileBins(self,data,fil="temp_dump_airnet_bins.txt"):
		t=time.time()
		top,bottom=np.array([2.00092774e+04,   1.00000000e+03,   1.00000000e+00, 9.15000000e+02,   6.81892407e+00,   9.15000000e+02, 1.23771376e+07]), np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.])
		print "splits calc: ", time.time()-t
		t=time.time()
		ws1,ws2 = self.init_reg.regressFromFileMultithreadedMultifileBins(data,top,bottom)
		with open(fil, 'a') as wrtr:
			wrtr.write("Lin Regression Weights\n")
			wrtr.write(str(ws1) + "\n" + str(ws2) + "\n")
		print "Lin Reg complete", time.time()-t
		self.w1, self.w2 = lm_opt_onepass_2targ_bins(ws1, ws2, (lambda wi1, wi2, datai: cross_entropy_error_from_multifile_multithreaded_bins(wi1, wi2, data,top,bottom)),data, 0.01)
		return self.w1, self.w2


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

def run_gowalla():
	open("temp_dump_gowalla.txt", 'w').close()
	er = ExternalLogisticRegressor()

	ws = er.regressFromFileMultithreadedMultiFile("temp/gowalla_ml_dataset")
	np.savetxt("fin_weights_gowalla.txt",ws)

def run_bin_airport():
	open("temp_dump_airnet_bins.txt", 'w').close()
	er = ExternalLogisticRegressor()

	ws = er.regressFromFileMultithreadedMultiFileBins("data/airport_net/dataset/airport_ds")
	with open("fin_weights_airnet_bins.txt", "w") as wrtr:
		wrtr.write(str(ws))



def run_airport():
	open("temp_dump_airnet.txt", 'w').close()
	er = ExternalLogisticRegressor()

	ws = er.regressFromFileMultithreadedMultiFile("data/airport_net/dataset/airport_ds")
	np.savetxt("fin_weights_airport.txt",ws)


# def easy_lin_regressor_unit_test():
# 	er = ExternalLogisticRegressor()
#
# 	ws = er.regressFromFileMultithreaded("rand_data_head.txt")
# 	#ws= er.regressFromFile("../../rand_xs.txt", "../../rand_ys.txt")
#
# 	dat = np.loadtxt("rand_data_head.txt")
# 	Xs = dat[:,:-1]
# 	ys = dat[:,-1]
#
# 	pca = PCA(11)
#
# 	Zs = pca.fit_transform(Xs)
#
# 	print ws
#
# 	plt.scatter(Zs[:,0], Zs[:,1], c=ys)
#
# 	zws = pca.transform([ws])
#
# 	delta = 0.025
# 	xsp = np.arange(-1.0, 1.0, delta)
# 	ysp = np.arange(-1.0, 1.0, delta)
# 	Xsp, Ysp = np.meshgrid(xsp, ysp)
# 	Zsp = zws[0,0]*Xsp + zws[0,1]*Ysp
#
# 	plt.contour(Xsp, Ysp, Zsp, [0])
#
# 	plt.show()
# 	#print calculate_class_error_from_file(Xs, ys, lambda x: er.classify(x))


#cProfile.run("easy_lin_regressor_unit_test()")
#cProfile.run("run()")
run_bin_airport()
#print get_split_norm_points("data/airport_net/dataset/airport_ds.txt", 212468368./50.)