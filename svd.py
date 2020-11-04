import csv
import numpy as np
import pandas as pd

class Recommender:
    
    k = 2
    dataset = []
    U = []
    M = []
    
    def __init__(self, k=2):
        self.k = k
    
    def read_data(self, filename="dataset.txt"):
        self.dataset = np.genfromtxt('dataset.txt', delimiter=',')

    def print_dataset(self):
        print(self.dataset)

    def get_factorization(self):

        return (self.U, self.M)

reco = Recommender()

reco.read_data()
reco.print_dataset()