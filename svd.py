import csv
import numpy as np
import pandas as pd

'''

Change in values:

v' = v + 2*learning_rate*error*others_factor

'''

songs = ['Loveyaatri','Channa mereya','Haan mein galat','Humnava mere','Akh lad Jaave','Shayad','Naah goriye','Bekhayali','Dil bechara','Tere liye duniya sajai mene','Dil Diya Gallan','Tum se hi','Kaise Hua','Mere Sohneya','Delhi se hai','Senorita - ZNMD','Kya baat hay','Matargashti','Ve maahi','Tum hi ho','Malang','Taroon ke shehar','Dil Ibaadat','Main rahoon ya na rahoon','Phir Se Udd Chala','Main tumhara','Dekhte dekhte','Kya muje pyar hai','Woh lamhein','Aashiq banaya aapne','Chumma chumma','Gulabi Aakhein','Bachna ee haseno','Nakhre','Bolna - Kapoor & sons','Khulke jeene ka','Illegal weapon 2.0','Khalibali','Proper patola','Pachtaoge','Ghunghroo','Odhani','Malhari','Garmi','Kar gayi chul','Naagin','She move it like','Jee karr da','Udd gaye-Ritviz','Zara zara','Mere Mehboob','Oo mere dil ke chen','Pehla nasha','Qafirana','Ye ratein ye mausam nadi ka kinara','Jeena jeena','Tu chahiye','Tu chahiye','Kaun tujhe yun pyar karega','Yaaro ne mere vaaste']

class Recommender:

	k = 4
	dataset = []
	U = []
	M = []
	learning_rate = 0.001
	epoches = 10000
	total_error = 0

	def __init__(self, k=4):
		self.k = k

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
		self.U = np.full((self.dataset.shape[0], self.k), 2.5)
		self.M = np.full((self.dataset.shape[1], self.k), 2.5)
		error = self.calc_error()
		print("Current Error is ", error)
		for e in range(self.epoches):
			if(e%100==0):
				# print("Current Error is ", error)
				pass
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
		return (self.U, self.M)
	
	def build_predicted(self):
		A = np.full(self.dataset.shape,0.0)
		for i in range(self.dataset.shape[0]):
			for j in range(self.dataset.shape[1]):
				p = self.U[i]
				q = self.M[j]
				er = 0
				for x in range(self.k):
					er += p[x]*q[x]
				A[i][j] = er
		np.savetxt("pred.csv", A, delimiter=",")
	
	def take_recommendation(self):
		print(self.dataset.shape)
		A = np.full((1,self.dataset.shape[1]), np.nan)
		for i in range(len(songs)):
			print(i+1, songs[i])
		x = int(input("Enter id of song you might like else 0 if don't want to add more songs: "))
		while(x!=0):
			r = input("How much will you rate"+songs[x-1]+" (1-5): ")
			A[0][x-1] = r
			for i in range(len(songs)):
				print(i+1, songs[i])
			x = int(input("Enter id of song you might like else 0 if don't want to add more songs: "))
		self.dataset = np.vstack((self.dataset,A))
		self.get_factorization()
		self.build_predicted()
		predict = []
		p = self.dataset.shape[0]-1
		for j in range(self.dataset.shape[1]):
			a = self.U[p]
			b = self.M[j]
			er = 0
			for kk in range(self.k):
				er += a[kk]*b[kk]
			if(j<len(songs)):
				predict.append([songs[j],er])
		predict = sorted(predict, key=lambda x: x[1])
		print(predict)

reco = Recommender()

reco.read_data("data2.csv")
reco.take_recommendation()