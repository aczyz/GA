import pandas as pd 
import aseeg as ag
import numpy as np
from sklearn.cross_decomposition import CCA

def filters(data):
    for i in range(len(data)):
        filterd1 = ag.pasmowozaporowy(data[i], 256, 49,51)
        filterd1 = ag.pasmowoprzepustowy(filterd1, 256, 5, 50)
        data[i] = (filterd1-filterd1.min())/(filterd1.max()-filterd1.min())
        return data

t1a = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_8Hz_Trial1_SUBJ1.csv", engine='python', header=None)
t1b = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_14Hz_Trial1_SUBJ1.csv", engine='python', header=None) 
t1s8hz = pd.DataFrame(t1a[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()
t1s14hz = pd.DataFrame(t1b[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()

t2a= pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_8Hz_Trial2_SUBJ1.csv", engine='python', header=None)
t2b = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_14Hz_Trial2_SUBJ1.csv", engine='python', header=None)
t2s8hz = pd.DataFrame(t2a[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()
t2s14hz = pd.DataFrame(t2b[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()

t3a = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_8Hz_Trial3_SUBJ1.csv", engine='python', header=None) 
t3b = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_14Hz_Trial3_SUBJ1.csv", engine='python', header=None)
t3s8hz = pd.DataFrame(t3a[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()
t3s14hz = pd.DataFrame(t3b[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()

t4a = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_8Hz_Trial4_SUBJ1.csv", engine='python', header=None) 
t4b = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_14Hz_Trial4_SUBJ1.csv", engine='python', header=None)
t4s8hz = pd.DataFrame(t4a[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()
t4s14hz = pd.DataFrame(t4b[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()

t5a = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_8Hz_Trial5_SUBJ1.csv", engine='python', header=None) 
t5b = pd.read_csv("C:/Users/aleks/Desktop/python/GA/SSVEP_14Hz_Trial5_SUBJ1.csv", engine='python', header=None)
t5s8hz = pd.DataFrame(t5a[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()
t5s14hz = pd.DataFrame(t5b[[15,17,21,23,28,30]][5*256:6*256].T).to_numpy()
random_pop=np.random.rand(int(input("How many targets? ")),256)
t=np.linspace(0,1,256)
sin14 = np.full((6,256), np.sin(2*np.pi*t*14))
sin8 = np.full((6,256), np.sin(2*np.pi*t*8))

signals = {"t1s14hz": filters(t1s14hz), "t1s8hz": filters(t1s8hz), "t2s8hz": filters(t2s8hz), "t2s14hz": filters(t2s14hz), "t3s8hz": filters(t3s8hz), "t3s14hz": filters(t3s14hz), "t4s8hz": filters(t4s8hz), "t4s14hz": filters(t4s14hz), "t5s8hz": filters(t5s8hz), "t5s14hz": filters(t5s14hz), "sin8": sin8,"sin14": sin14, "random": random_pop}

