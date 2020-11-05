import csv
import numpy as np

'''

Change in values:

v' = v + 2*learning_rate*error*others_factor

'''

songs = ['Loveyaatri','Channa mereya','Yaaro ne mere vaaste','Haan mein galat','Humnava mere','Akh lad Jaave','Shayad','Naah goriye','Bekhayali','Dil bechara','Tere liye duniya sajai mene','Dil Diya Gallan','Tum se hi','Kaise Hua','Mere Sohneya','Delhi se hai','Senorita - ZNMD','Kya baat hay','Matargashti','Ve maahi','Tum hi ho','Malang','Taroon ke shehar','Dil Ibaadat','Main rahoon ya na rahoon','Phir Se Udd Chala','Main tumhara','Dekhte dekhte','Kya muje pyar hai','Woh lamhein','Aashiq banaya aapne','Chumma chumma','Gulabi Aakhein','Bachna ee haseno','Nakhre','Bolna - Kapoor & sons','Khulke jeene ka','Illegal weapon 2.0','Khalibali','Proper patola','Pachtaoge','Ghunghroo','Odhani','Malhari','Garmi','Kar gayi chul','Naagin','She move it like','Jee karr da','Udd gaye-Ritviz','Zara zara','Mere Mehboob','Oo mere dil ke chen','Pehla nasha','Qafirana','Ye ratein ye mausam nadi ka kinara','Jeena jeena','Tu chahiye','Kaun tujhe yun pyar karega']

class Recommender:

	k = 5
	read = True
	dataset = []
	U = []
	M = []
	learning_rate = 0.001
	epoches = 1000
	total_error = 0

	def __init__(self, k=5, read=True):
		self.k = k
		self.read = read

	def read_data(self, filename="dataset.txt"):
		self.dataset = np.genfromtxt(filename, delimiter=',')

	def print_dataset(self):
		print(self.dataset)

	def error(self, u, v):
		return float(abs(u-v)**2)

	def calc_error(self):
		error = 0
		for i in range(self.dataset.shape[0]):
			for j in range(self.dataset.shape[1]):
				if(not np.isnan(self.dataset[i][j])):
					p = self.U[i]
					q = self.M[j]
					er = 0
					for x in range(self.k):
						er += p[x]*q[x]
					error += self.error(er, self.dataset[i][j])
		self.total_error = error
		return error

	def get_factorization(self):
		if(self.read):
			self.U = np.genfromtxt("U.csv", delimiter=",")
			self.M = np.genfromtxt("M.csv", delimiter=",")
		else:
			self.U = np.random.rand(self.dataset.shape[0], self.k)
			self.M = np.random.rand(self.dataset.shape[1], self.k)
			error = self.calc_error()
			for e in range(self.epoches):
				if(e%100==0):
					print("Current Error is ", error)
					# pass
				if(self.total_error-error<=0.01 and e!=0):
					break
				self.total_error = error
				error = 0
				for i in range(self.dataset.shape[0]):
					for j in range(self.dataset.shape[1]):
						if(not np.isnan(self.dataset[i][j])):
							p = self.U[i]
							q = self.M[j]
							er = 0
							for x in range(self.k):
								er += p[x]*q[x]
							error += self.error(er, self.dataset[i][j])
							er = self.dataset[i][j] - er
							for x in range(self.k):
								to_add_u = 2*self.learning_rate*er*self.M[j][x]
								to_add_m = 2*self.learning_rate*er*self.U[i][x]
								self.U[i][x] += to_add_u
								self.M[j][x] += to_add_m
			np.savetxt("U.csv", self.U, delimiter=",")
			np.savetxt("M.csv", self.M, delimiter=",")
		return (self.U, self.M)
	
	def build_predicted(self, add=1):
		A = np.full((self.dataset.shape[0]+add,self.dataset.shape[1]),0.0)
		for i in range(self.dataset.shape[0]+add):
			for j in range(self.dataset.shape[1]):
				p = self.U[i]
				q = self.M[j]
				er = 0
				for x in range(self.k):
					er += p[x]*q[x]
				A[i][j] = er
		np.savetxt("pred.csv", A, delimiter=",")
		return A
	
	def add_reco(self, A, M, r):
		for i in range(len(A)):
			A[i] = A[i] + (r-2.5)*M[i]
		return A
	
	def take_recommendation(self):
		self.get_factorization()
		A = np.full((self.k), 0.0)
		for i in range(len(songs)):
			print(i+1, songs[i])
		x = int(input("Enter id of song you might like else 0 if don't want to add more songs: "))
		while(x!=0):
			r = input("How much will you rate"+songs[x-1]+" (1-5): ")
			A = self.add_reco(A, self.M[x-1], float(r))
			print(A)
			# for i in range(len(songs)):
			# 	print(i+1, songs[i])
			x = int(input("Enter id of song you might like else 0 if don't want to add more songs: "))
		predict = []
		for j in range(self.dataset.shape[1]):
			b = self.M[j]
			er = 0
			for kk in range(self.k):
				er += A[kk]*b[kk]
			if(j<len(songs)):
				predict.append([songs[j],er])
		predict = sorted(predict, key=lambda x: x[1])
		predict = [*reversed(predict)]
		for i in range(10):
			print(i+1, predict[i][0],": ",predict[i][1])
	
	def take_recommendation_from_one(self):
		self.get_factorization()
		for i in range(len(songs)):
			print(i+1, songs[i])
		x = int(input("Enter id of song you like: "))
		A = self.M[x-1]
		predict = []
		for j in range(self.dataset.shape[1]):
			b = self.M[j]
			er = 0
			for kk in range(self.k):
				er += A[kk]*b[kk]
			if(j<len(songs) and j != x-1):
				predict.append([songs[j],er])
		predict = sorted(predict, key=lambda x: x[1])
		predict = [*reversed(predict)]
		for i in range(10):
			print(i+1, predict[i][0],": ",predict[i][1])

def main():

	ans = input("Do you want to recalculate the matrix factorization(Y/N): ")
	reco = None
	if(ans=="Y"):
		reco = Recommender(read=False)
	else:
		reco = Recommender()

	reco.read_data("Form Data.csv")

	ans = int(input("Do you want to get song recommendation from 1 song or you want to get it from rating multiple songs (1/2): "))

	if(ans==1):
		reco.take_recommendation_from_one()
	else:
		reco.take_recommendation()

if __name__ == "__main__":
	main()