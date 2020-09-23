import numpy as np
import matplotlib.pyplot as plt
import random
import aseeg as ag
import math
import datasets as ds
from sklearn.cross_decomposition import CCA
from collections import OrderedDict


class GA_cca():
    def __init__(self, population, data1, data2, data1a, data2a):
        self.data1 = data1
        self.data2 = data2
        self.data1a = data1a
        self.data2a = data2a
        self.population = population
        self.genes=ds.targets
        self.pop_length=len(population)
        self.cross_p=0.7
        self.mutation_p=0.1
        self.fits = {}
        self.fits_values=[]

    def cca_cor(self, signal, pop):
        n_components = 1
        cca = CCA(n_components)
        cca.fit(signal,pop)
        U, V = cca.transform(signal,pop)
        return abs(np.corrcoef(U.T, V.T)[0, 1])

    #fitness - one target (1x256)
    def fitness(self):
        for i in range(self.pop_length):
            cor1 = self.cca_cor(self.data1.T, self.population[i])
            cor1a = self.cca_cor(self.data1a.T, self.population[i])
            cor2 = self.cca_cor(self.data2.T, self.population[i])
            cor2a = self.cca_cor(self.data2a.T, self.population[i])

            # self.fits[i]=round(math.sqrt(((1-cor1)+cor1a)**2) + (cor2+cor2a)**2, 3)
            self.fits[i]=round(math.sqrt((cor1-1)**2+cor2**2)+math.sqrt((cor1a-1)**2+cor2a**2),3)


        best_key = min(self.fits, key=self.fits.get)
        best_value = self.fits[best_key]
        best_target = self.population[best_key]

        return (best_target, best_value)

        # tournament selection -> takes some targets from pop and adds best to new pop (same size)
    def tournament(self):
        self.fits_values=list(self.fits.values())
        fits_keys=list(self.fits.keys())
        new_Population=self.population[:]
        for i in range(self.pop_length):
            duel_fits = random.sample(fits_keys, (int(0.4*(self.pop_length))))
            duel = list(self.fits_values[j] for j in duel_fits)
            idx_win = duel.index(min(duel))
            new_Population[i] = self.population[duel_fits[idx_win]]


        self.population=new_Population
        return self.population

# crossover probability - 0.7
    def crossing_over(self):
        rand_cross=np.random.rand(1,math.floor(self.pop_length/2))
        new_Population = self.population
        for i in range(0,self.pop_length-1,2):
            if rand_cross[0][int(i/2)]<=self.cross_p:
                cross_loc=np.random.randint(1,self.genes-1)

                child1 =(np.concatenate((((new_Population[i])[:cross_loc]),((new_Population[i+1])[cross_loc:]))))
                child2 =(np.concatenate((((new_Population[i+1])[:cross_loc]),((new_Population[i])[cross_loc:]))))

                new_Population[i] = child1
                new_Population[i+1] = child2

        self.population = new_Population
        return self.population

# mutation probability - 0.1
    def mutation(self):
        new_Population=self.population[:]
        rand_mut=np.random.rand(1,self.pop_length)
        for i in range(self.pop_length):
            if rand_mut[0][i]<=self.mutation_p:
                mut_loc=np.random.randint(0,255)
                new_Population[i][mut_loc]="%.2f" % (random.uniform(0, 1))

        self.population = new_Population
        return self.population

    def main(self):
        everyten = []
        gen_count = 0
        gen = 100
        self.best_values = []
        while gen_count < gen:
            gen_count+=1
            #best_target = self.fitness()[0]
            self.best_values.append(self.fitness()[1])
            self.population = self.tournament()
            self.population = self.crossing_over()
            self.population = self.mutation()
            if (gen_count % 10) == 9:
                everyten.append()
                print(np.mean(self.fits_values))
        return self.fitness()[0]

    def plots(self):
        plt.subplot(211)
        plt.plot(self.best_values)
        plt.title("fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness value")
        plt.subplot(212)
        plt.plot(self.fitness()[0])
        #plt.plot(self.best_targets[len(self.best_targets)-1])
        #plt.plot(self.population[len(self.population)-1])
        plt.title("Best target")
        plt.show()

#to run single GA (500 gen) - uncomment following commands (arguments: initial population, signal to resemble, signal to differenciate)
#w = GA(ds.randpop(), ds.choose_data('2 2 14'), ds.choose_data('2 2 8'), ds.sin14, ds.sin8, ds.cos14, ds.cos8)
#w.main()