class User:

		def __init__(self,uid,comm,pos,friends):
				self.uid=uid
				self.comm=comm
				self.pos=pos
				self.friends=friends
				self.deg1=len(friends)
				self.deg2=None

		def __repr__(self):
			return str(self)

		def __str__(self):
			return str(self.uid) + ": " + str(self.friends)
