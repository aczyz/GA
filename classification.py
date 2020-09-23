import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import aseeg as ag
import scipy.fftpack as sf
import math
import datasets as ds
from Gen_Alg import GA
from GA_cca import GA_cca
from sklearn.cross_decomposition import CCA

#tworzy plik txt
results = open("datar41_sig.txt","w")

#funkcja obliczająca korelację (Pearson). Wpisuje do pliku txt wynik w formie: "korelacja wyższa", współczynnik 1, współczynnik 2
def compare(signal, ga1, ga2):
    for d in range(6):
        cor1 = str(abs(round(np.corrcoef(signal[d], ga1)[0,1], 3)))
        cor2 = str(abs(round(np.corrcoef(signal[d], ga2)[0,1], 3)))
        if cor1 > cor2:
            results.writelines(" ".join(["\n cor1", cor1, cor2]))
        else:
            results.writelines(" ".join(["\n cor2", cor1, cor2]))

#zwraca rezultat CCA
def cca_cor(signal, pop):
    n_components = 1
    cca = CCA(n_components)
    cca.fit(signal,pop)
    U, V = cca.transform(signal,pop)
    return abs(np.corrcoef(U.T, V.T)[0, 1])

#funkcja obliczająca korelację (CCA), rezultat zapisywany do pliku analogicznie
def compares(signal, ga1, ga2, sin1, sin2, cos1, cos2):
    for d in range(6):
        cor1 = cca_cor(signal.T, ga1)
        corsin1 = cca_cor(signal.T, sin1[d])
        corcos1 = cca_cor(signal.T, cos1[d])
        cor2 = cca_cor(signal.T, ga2)
        corsin2 = cca_cor(signal.T, sin2[d])
        corcos2 = cca_cor(signal.T, cos2[d])

        # cor1a=max(corsin1, corcos1)
        # cor2a=max(corsin2,corcos2)
        # cors1 = cor1 + cor1a
        # cors2 = cor2 + cor2a
        cors1=round((math.sqrt(cor1-1)**2+(cor2)**2)+(math.sqrt(corsin1-1)**2+(corsin2)**2)+(math.sqrt(corcos1-1)**2+(corcos2)**2),3)
        cors2=round((math.sqrt(cor2-1)**2+(cor1)**2)+(math.sqrt(corsin2-1)**2+(corsin1)**2)+(math.sqrt(corcos2-1)**2+(corcos1)**2),3)

        if cors1 > cors2:
            results.writelines(" ".join(["cor1", str(cors1), str(cors2), "\n"]))
        else:
            results.writelines(" ".join(["cor2", str(cors1), str(cors2), "\n"]))

#wersja niżej dostosowana do algorytmu GA opartego na CCA

#funkcja generująca zestaw najlepszych osobników dla wybranej osoby (k) ze wszystkich triali i zwraca je w formie pary list (osobniki dla 8Hz, osobniki dla 14Hz)
def genalgs(k):
    gens8 = []
    gens14 = []
    for j in range(1,6):
        sn = '{} {} {}'.format(k,j,8)
        cn = '{} {} {}'.format(k,j,14)
        inst = GA_cca(ds.choose_data("1 1 8"), ds.choose_data(sn), ds.choose_data(cn), ds.sin8, ds.sin14, ds.cos8, ds.cos14)
        gens8.append(inst.main())
        inst2 = GA_cca(ds.choose_data("1 1 14"), ds.choose_data(cn), ds.choose_data(sn), ds.sin14, ds.sin8, ds.cos14, ds.cos8)
        gens14.append(inst2.main())
    return (gens8, gens14)

gen_results =  genalgs(1)

#funkcja zarządzająca klasyfikacją dla wybranej osoby (i). Porównuje wszystkie triale danej osoby z rezultatami działania GA (14Hz i 8Hz) i zapisuje do pliku txt wynik klasyfikacji. Klasyfikacja z użyciem CCA - funkcja compares, zby klasyfikować z użyciem Pearsona trzeba zmienić na funkcję compare.
def doitforme(i):
    for j in range(1, 6):
        for k in [14, 8]:
            ch = '{} {} {}'.format(i,j,k)
            for l in range(5):
                compares(ds.choose_data(ch), (gen_results[0])[l], (gen_results[1])[l], ds.sin8, ds.sin14, ds.cos8, ds.cos14)
                results.writelines(" ".join(["\n", ch, str(l), "\n"]))
    results.writelines("DONE 41\n")

doitforme(1)
