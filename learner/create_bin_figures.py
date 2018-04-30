import numpy as np
import matplotlib.pyplot as plt


bins=[0,50,100,200,400,800,1600,3200,6400,12800,20000]

weights = np.array([[  8.78926839e+01,  -5.36662589e+01,  -5.81099823e-03,
          5.67057845e-02,   8.59323825e-02,  -4.45459988e-02,
         -5.11839029e-02],
       [  1.05498428e+02,  -5.56102702e+01,   2.92521537e-02,
          1.65750739e-01,   1.25844149e-02,  -7.94564373e-02,
         -1.70289167e-01],
       [  1.76623893e+02,  -7.00857746e+01,   1.73377344e-02,
         -1.04190379e-02,   7.05761031e-02,  -6.43393079e-02,
         -4.00806450e-02],
       [ -2.42309889e+02,   5.10245937e+01,  -1.42574985e-01,
          1.80835705e-01,  -1.72362590e-01,  -1.79150340e-01,
         -5.34399510e-02],
       [ -5.68492892e+01,   4.78113332e+00,  -2.56946623e-02,
          9.49806271e-02,  -3.48187102e-02,  -9.31545735e-02,
         -6.70929662e-02],
       [ -3.75562912e+01,   2.22616665e+00,  -3.78444729e-02,
          1.35836662e-01,  -1.24493719e-01,  -1.79939617e-01,
         -6.12491267e-02],
       [  6.54035690e-01,  -1.01765890e+00,   6.22580387e-03,
          6.45201051e-02,   2.12733357e-02,  -6.86863347e-02,
         -8.16389238e-02],
       [  1.68574633e-02,  -9.32903816e-01,   3.52811793e-03,
          6.41435750e-02,   2.99453255e-02,  -5.18712809e-02,
         -8.52165390e-02],
       [ -8.42452736e-03,  -9.17614884e-01,   6.20191707e-03,
          5.35598600e-02,   2.85930510e-02,  -7.58857786e-02,
         -7.96850208e-02],
       [  2.09235049e-02,  -9.33904718e-01,  -1.32675092e-02,
          6.05719911e-02,   2.35251192e-02,  -7.24406912e-02,
         -9.89141133e-02]])


weights=abs(weights/np.sum(weights,axis=0))

norm_weights=np.zeros(weights.shape)
for i in range(weights.shape[0]):
	norm_weights[i] = weights[i]/np.max(weights[i])
weights=norm_weights
N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.1       # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(ind, weights[:,0], width, color='r')
rects2 = ax.bar(ind+width, weights[:,1], width, color='pink')
rects3 = ax.bar(ind+2*width, weights[:,2], width, color='blue')
rects4 = ax.bar(ind+3*width, weights[:,3], width, color='g')
rects5 = ax.bar(ind+4*width, weights[:,4], width, color='y')
rects6 = ax.bar(ind+5*width, weights[:,5], width, color='black')
rects7 = ax.bar(ind+6*width, weights[:,6], width, color='grey')

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]), ('Distance', 'Weighted Distance', 'Communities in Common', 'Degree', 'Log Degree', 'Median Routing Power', 'Locality'))

# add some text for labels, title and axes ticks
ax.set_ylabel('Weight (Normalized)')
ax.set_xlabel('Bin Lower Bound (km)')
ax.set_title('Binned Weigths')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(map(str,bins))




plt.show()