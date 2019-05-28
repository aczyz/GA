import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import aseeg as ag
import scipy.fftpack as sf
import math
from sklearn.cross_decomposition import CCA
from collections import OrderedDict
import trials_data as td


best_t = np.zeros((256,6))
class Gen_Algorithm():
    def __init__(self):
        #self.genes=int(input("How many genes in a target? "))
        self.genes=256
        #self.pop_length=int(input("How many targets in the population? "))
        self.pop_length=6
        self.current_pop=td.signals[input("Initial population: ")]
       
        self.data1=td.signals[input("Choose data1: ")]
        self.data2=td.signals[input("Choose data2: ")]
        n_components = 1
        self.cca = CCA(n_components)

        self.best_fits=[]
        self.fits={}
        self.draw=np.random.rand(1,self.pop_length)
        self.cross_p=0.7
        self.mutation_p=0.1
        self.max_fit=0
        self.sum_fit=0

#sqrt((1-corr8)^2+corr14^2)
        
    def best_target(self):
        global best_t
        for i in range(len(self.current_pop)):
            self.cca.fit(self.data1.T, self.current_pop[i])
            U, V = self.cca.transform(self.data1.T, self.current_pop[i])
            self.cca.fit(self.data2.T, self.current_pop[i])
            M, N = self.cca.transform(self.data2.T, self.current_pop[i])
            self.fits[i]=math.sqrt((1-abs(np.corrcoef(U.T, V.T)[0, 1]))**2 + abs(np.corrcoef(M.T, N.T)[0, 1])**2)
        self.sum_fit=sum(self.fits.values())
        self.best=max(self.fits, key=self.fits.get)
        self.max_fit=self.fits[self.best] 
        self.best_fits.append(self.max_fit)
        best_t = self.current_pop[self.best]
        print("fitness value: ", self.max_fit)


    def roulette(self):
        self.fits=OrderedDict(sorted(self.fits.items(),
        key=lambda x: x[1], reverse=True))
        fits_list=list(self.fits)
        new_Population=self.current_pop[:]
        for i in range(int(self.pop_length/2)):
            new_Population[i]=self.current_pop[fits_list[i]]
            new_Population[i+int(self.pop_length/2)]=self.current_pop[fits_list[i]]

        self.current_pop=new_Population
        
    def crossing_over(self):
        rand_cross=np.random.rand(1,math.floor(self.pop_length/2)) 

        for i in range(0,self.pop_length-1,2):
            if rand_cross[0][int(i/2)]<=self.cross_p:
                cross_loc=np.random.randint(1,self.genes-1)
                self.current_pop[i]=(np.concatenate((((self.current_pop[i])[:cross_loc]),((self.current_pop[i+1])[cross_loc:]))))
                self.current_pop[i+1]=(np.concatenate((((self.current_pop[i+1])[:cross_loc]),((self.current_pop[i])[cross_loc:]))))

    def mutation(self):
        rand_mut=np.random.rand(1,self.pop_length)

        for i in range(self.pop_length):
            if rand_mut[0][i]<=self.mutation_p: 
                mut_loc=np.random.randint(0,self.genes-1) 
                (self.current_pop[i])[mut_loc]="%.8f" % (random.uniform(0, 1)) 

    def plots(self):
        global best_t
        plt.subplot(311)
        plt.plot(sf.fft(best_t))
        plt.ylim([0,40])
        plt.xlim([0,20])
        plt.subplot(312)
        plt.plot(best_t)
        plt.subplot(313)
        plt.plot(self.best_fits)
        plt.show()
                     
def main():
    m=Gen_Algorithm()
    gen_count=0
    gens=int(input("How many generations? "))
    while m.max_fit < 8 and gen_count < gens:
        gen_count+=1
        m.best_target()
        m.roulette()
        m.crossing_over()
        m.mutation()
    print("Best target: ",  best_t)
    m.plots()
    print("Amount of generations: ", gen_count)
        
main()



