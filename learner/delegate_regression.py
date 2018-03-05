import sys
import numpy as np


def delegate_cross_entropy_error_from_file(w, f, lim, out):
	se=0
	sg=0
	sh=0
	count=0
	line=f.readline()
	while count<lim and line:
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
	outse
	ret[pnum]=[se, sg, sh]

init_weights_file = sys.argv[1]

lstart = sys.argv[3]
lend = sys.argv[4]

np.loadtxt(init_weights_file)