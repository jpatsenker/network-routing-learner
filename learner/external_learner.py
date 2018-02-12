import numpy as np

class ExternalRegression:

	def __init__(self):
		self.xtx = 0
		self.xty = 0

	def addPoint(self, x, y):
		"""
		adds point to the regressor
		:param x: list of features
		:param y: label
		"""
		self.xtx += np.dot(np.reshape(x,[1,len(x)]),np.reshape(x,[len(x),1]))
		self.xty += y*x

	def final(self):
		return np.dot(np.linalg.pinv(self.xtx),np.reshape(self.xty, [len(self.xty),1]))

	def regressFromFile(self,data,labels):
		with open(data) as f:
			with open(labels) as g:
				x=f.readline()
				y=g.readline()
				while x:
					self.addPoint(np.fromstring(x, dtype=float, sep=','), int(y))
					x=f.readline()
					y=g.readline()
		return self.final()


