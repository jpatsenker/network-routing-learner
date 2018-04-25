import pandas as pd
import numpy as np
from core.user import User

a=pd.read_csv("data/airport_net/airports.dat")
airmat=a.as_matrix()
good_ids = airmat[:-1,4]
bad_ids = airmat[:-1,5]
good_pos = airmat[:-1,6:8].astype(float)

r = pd.read_csv("data/airport_net/routes.dat")
routmat = r.as_matrix()

gil = good_ids.tolist()
# id_map = {}
# for i in range(len(gil)):
# 	tok=gil[i]
# 	if type(gil[i]) == float or gil[i]=='\\N':
# 		tok=bad_ids[i]
# 	id_map[tok]=i

an = {}
for i in range(len(good_ids)):
	tok=gil[i]
	if type(gil[i]) == float or gil[i]=='\\N':
		tok=bad_ids[i]
	friends1=np.where(routmat[:,0]==tok)[0]
	friends2=np.where(routmat[:,1]==tok)[0]
	u = User(i,[],(good_pos[i,0],good_pos[i,1]),[])
	u.friends.extend(friends1)
	u.friends.extend(friends2)
	if i%1000==0:
		print(i)

import pickle

with open('data/airport_net/airnet.pkl','wb') as w:
	pickle.dump(an,w)