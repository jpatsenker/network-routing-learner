import sys
import numpy as np


def delegate_cross_entropy_error_from_file(w, f, lim):
	se=0
	sg=0
	sh=0
	count=0
	line=f.readline()
	while count<lim and line:
		count+=1.
		line=np.fromstring(line,dtype=float,sep=' ')
		x=line[:-1]
		y1=line[-1]
		y2=line[-2]
		exppart=np.e**(y*np.dot(w,x))
		part2 = -y/(1.+exppart)
		invc=1./count
		se += (np.log(1. + 1./exppart)-se)*invc
		sg += (x*part2-sg)*invc
		sh += (-part2**2*np.outer(x,x)*exppart-sh)*invc
		line=f.readline()
	return se, sg, sh
init_weights_file = sys.argv[1]

input_file = sys.argv[2]

output_file = sys.argv[3]

lstart = float(sys.argv[4])
lend = float(sys.argv[5])

iw = np.loadtxt(init_weights_file)

file_read = open(input_file)

c=0
while c<lstart:
	file_read.readline()
	c+=1

se, sg, sh= delegate_cross_entropy_error_from_file(iw, file_read, lend-lstart)

file_read.close()

np.savetxt(weights, )