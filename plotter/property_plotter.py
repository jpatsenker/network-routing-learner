import sys
sys.path.append("../..")

from gowalla_research.core.user import User
from gowalla_research.core.connection import Edge


import matplotlib.pyplot as plt
import numpy as np


colors = ['r', 'g', 'b', 'y']

class PropertyPlotter:

	def __init__(self, gnames, pnames, pdatax, pdatay):
		self.gnames = gnames
		self.pnames = pnames
		self.pdatax = pdatax
		self.pdatay = pdatay
		for i in range(len(pnames)):
			plt.plot(pdatax, map(lambda x: x[i], pdatay), color=colors[i])
		ax = plt.axes()
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, pnames)

	def showPlot(self):
		plt.show()

	def savePlot(self,name):
		plt.savefig(name)	
