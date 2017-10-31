import sys
sys.path.append("../.")

import cPickle as pickle
from plotter.property_plotter import PropertyPlotter


data_name="full_random/results_singular/FINAL10MORE.pkl"
plot_dir = "plots/final100/"

data = pickle.load(open(data_name, "r"))


graphnames = map(str, range(1000,11000,1000))





p = PropertyPlotter(graphnames, ["equal", "distance", "communities", "degree"], range(1000,11000,1000), data["s"])
p.savePlot(plot_dir + "Subtraction10MORE.png")