import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("evaluations/air_aw.txt")


plt.hist(data,bins=25)
plt.show()