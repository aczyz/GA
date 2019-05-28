import ga_v1a as ga1
import ga_v2 as ga2
import ga_v3 as ga3
import ga_v4 as ga4
import ga_v5 as ga5
import ga_v6 as ga6
from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd
import aseeg as ag
import math
import trials_data as td

#compares2(td.signals["t1s8hz"],td.signals["t1s14hz"],a)
a = input("Choose GA version ").best_t


def compares(signal1,signal2):
    cors8hz=abs(np.corrcoef(a.T, signal1.T)[0, 1])
    cors14hz=abs(np.corrcoef(a.T, signal2.T)[0, 1])
    if cors14hz > cors8hz: 
        print("Correlates more with 14hz (","%.3f" % cors14hz,")\nthan with 8hz (","%.3f" % cors8hz, ")")
    else:
        print("Correlates more with 8z (", "%.3f" % cors8hz,")\nthan with 14hz (","%.3f" % cors14hz, ")")


def compares2(signal1,signal2,a):
    n_components = 1
    cca = CCA(n_components)
    cca.fit(signal1.T, a)
    U, V = cca.transform(signal1.T, a)
    cca.fit(signal2.T, a)
    M, N = cca.transform(signal2.T, a)
    cor1=(abs(np.corrcoef(U.T, V.T)[0, 1])) 
    cor2=(abs(np.corrcoef(M.T, N.T)[0, 1]))
    if cor1 > cor2: 
        print("Correlates more with 14hz (","%.3f" % cor1,")\nthan with 8hz (","%.3f" % cor2, ")")
    else:
        print("Correlates more with 8z (", "%.3f" % cor2,")\nthan with 14hz (","%.3f" % cor1, ")")