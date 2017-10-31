import sys
import math

sys.path.append("..")

import numpy as np
from sklearn.linear_model import LogisticRegression
from plotter.funcplotter2D import FuncPlotter2D

class Learner:
	@staticmethod
	def COLS():
		return 1
	@staticmethod
	def ROWS():
		return 0

	def __init__(self, training_xs, training_ys):
		self.training_xs = training_xs
		self.training_ys = training_ys
		self.weights = [0 for i in range(len(training_xs[0]))]

	def learn(self):
		print "FUNCTION NOT IMPLEMENTED!!!"

	def hypfunc(self, x):
		if len(x) != len(self.weights):
			print "INCORRECT X DIMENSIONS"
			#print x
			#print self.weights
			return

		return reduce(lambda s, i: self.weights[i]*x[i] + s, range(len(x)), 0)

	def hypfunc_class(self,x):
		res = self.hypfunc(x)
		if res >= 0:
			return 1
		return 0

	def hypfunc_log(self,x):
		return 1.0/(1+math.e**(-self.hypfunc(x)))

class LogisticRegressionLearner(Learner, object):
	#CHECK SANITY OF LOGISTIC REGRESSION
	def __init__(self, txs, tys):
		super(LogisticRegressionLearner, self).__init__(txs, tys)
		self.training_ys = np.array(self.training_ys)
		self.training_xs = np.matrix(self.training_xs)

	def learn(self):
		l = LogisticRegression()
		l.fit(self.training_xs, self.training_ys)
		self.weights = l.coef_[0]

	def test(self, test_xs, test_ys):
		predicted_xs = map(lambda x: self.hypfunc_log(x), (test_xs))





# tx = [[1,1,1],[1,0,1],[1,1,0]]
# ty = [1, -1, -1]
# l = LogisticRegressionLearner(tx, ty)
#
# l.learn()
# print l.hypfunc([1,1,1])
# print l.hypfunc([1,0,1])
# print l.hypfunc([1,1,0])
# print l.hypfunc([1,-1,-1])
# print l.hypfunc([1,1.75,0])
#
# f = FuncPlotter2D((-2,2),(-2,2), .05)
# f.addFuncToPlot(lambda x,y: l.hypfunc([1,x,y]))
# f.addPoints([[1,1]],"blue")
# f.addPoints([[0,1],[1,0]],"red")
#
# f.showPlot()