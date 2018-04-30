import numpy as np
import matplotlib.pyplot as plt


data1 = np.loadtxt("evaluations/aa.txt")
data2 = np.loadtxt("evaluations/ag.txt")
data3 = np.loadtxt("evaluations/ga.txt")
data4 = np.loadtxt("evaluations/gg.txt")
#
# bins = np.linspace(0, 19, 10)
#
# fig = plt.figure()
#
#
#
# plt.subplot(411)
# plt.title('Airport Net with Airport Weights')
# plt.hist(data1,bins,color='r')
# plt.subplot(412)
# plt.title('Airport Net with Gowalla Weights')
# plt.hist(data2,bins,color='g')
# plt.subplot(413)
# plt.title('Gowalla Net with Airport Weights')
# plt.hist(data3,bins,color='b')
# plt.subplot(414)
# plt.title('Gowalla Net with Gowalla Weights')
# plt.hist(data4,bins,color='y')
# plt.xlabel('Stretch')
# plt.tight_layout()
# plt.show()
#
#


print np.median(data1),np.median(data2),np.median(data3),np.median(data4)