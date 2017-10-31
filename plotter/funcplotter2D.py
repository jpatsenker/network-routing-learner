__author__ = 'jpatsenker'

import sys
sys.path.append("../..")


import matplotlib.pyplot as plt
import numpy as np


colors = ['r', 'g', 'b', 'y']

class FuncPlotter2D:

	def __init__(self, xlims, ylims, resolution):
		self.xlims = xlims
		self.ylims = ylims
		self.resolution = resolution
		ax = plt.axes()
		ax.grid(True, which='both')

	def addFuncToPlot(self, f):
		delta = self.resolution
		xrange = np.arange(self.xlims[0], self.xlims[1], delta)
		yrange = np.arange(self.ylims[0], self.ylims[1], delta)
		X, Y = np.meshgrid(xrange,yrange)
		plt.contour(X, Y, f(X,Y), [0])

	def addPoints(self, points, color):
		plt.plot([row[0] for row in points],[row[1] for row in points], 'ro', color=color)

	def showPlot(self):
		plt.show()

	def savePlot(self,name):
		plt.savefig(name)


# f = FuncPlotter2D((-1,1),(-1,1), .05)
# f.addFuncToPlot(lambda x,y: x**2+y)
# f.showPlot()