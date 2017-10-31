from user import User
import math

def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

class Edge:

		def __init__(self, eid, uid1, uid2):
				self.eid = eid
				self.uid1 = uid1
				self.uid2 = uid2

		def fill_in_data(self,users):
				u1= users[self.uid1]
				u2 = users[self.uid2]
		#print u1.uid
				self.dist = distance(u1.pos, u2.pos)
				self.mutual = len(set.intersection(set(u1.friends),set(u2.friends)))
				self.commoncom = len(set.intersection(set(u1.comm),set(u2.comm)))
				self.rp1 = u2.deg1
				self.rp2 = u2.deg2

		def __str__(self):
				return str(self.uid1) + "---" + str(self.uid2)

		def __repr__(self):
				return self.__str__()



