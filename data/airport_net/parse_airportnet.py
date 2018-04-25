from copy import copy
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.getcwd())
from core.user import User

a=pd.read_csv("data/airport_net/airports.dat")
airmat=a.as_matrix()
good_ids = airmat[:-1,0].astype(str)
good_pos = airmat[:-1,6:8].astype(float)

r = pd.read_csv("data/airport_net/routes.dat")
routmat = r.as_matrix()

a=np.stack([routmat[:,3],routmat[:,5]]).T
a = np.delete(a, np.where(a=='\\N')[0], axis=0)
#np.savetxt("route_diluted.txt", a.astype(str), fmt='%s')


an={}
for i in range(len(good_ids)):
	friends1=a[np.where(a[:,0]==good_ids[i])[0],1]
	friends2=a[np.where(a[:,1]==good_ids[i])[0],0]
	real_friends = []
	for f in friends1:
		if f in good_ids:
			real_friends.append(f)
	for f in friends2:
		if f in good_ids and f not in real_friends:
			real_friends.append(f)
	u = User(good_ids[i],[],(good_pos[i,0],good_pos[i,1]),real_friends)
	if i%1000==0:
		print(i)
	an[good_ids[i]]=u


an_new=copy(an)
for port in an:
	if len(an[port].friends)==0:
		an_new.pop(port)

an=an_new

import pickle

with open('data/airport_net/airnet.pkl','wb') as w:
	pickle.dump(an,w)