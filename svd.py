import csv
import numpy as np
import pandas as pd


class Recommender:

	k = 2
	dataset = []
	U = []
	M = []
	learning_rate = 0.001
	epoches = 0
	total_error = 0

	def __init__(self, k=2):
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
		return (self.U, self.M)


reco = Recommender()

reco.read_data()
reco.print_dataset()
print(reco.get_factorization())
print(reco.calc_error())