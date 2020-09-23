import aseeg as ag
import numpy as np
import random
import matplotlib.pyplot as plt

targets=100
#signal normalization
def normalize(data):
    minS = data.min()
    maxS = data.max()
    signal = (data-minS)/(maxS-minS)
    return signal

# filters and normalizes data + rounds numbers to 2 decimal places
def prepareData(data):
    for i in range(len(data)):
        filter1 = ag.pasmowozaporowy(data[i], 256, 49,51)
        filter2 = ag.pasmowoprzepustowy(filter1, 256, 5,30)
        data[i] = normalize(filter2)
    return np.round(data,2)

# chooses data from datasets files -> needs 'a b c' input
# a-> subject, b-> trial, c-> Hz
def choose_data(choice):
    choice = list(choice.split(" "))
    electrodes = [14,20,22,27]
    # file_path = "C:/Users/Marcin/Dropbox/SSVEP/Bakardjian/"
    file_path = "C:/Users/aleks/desktop/python/GA/daneSSVEP/"
    a = int(choice[0])
    b = int(choice[1])
    c = int(choice[2])
    filename = 'Subj{}/SSVEP_{}Hz_Trial{}_SUBJ{}.csv'.format(a,c,b,a)
    raw_signal = np.genfromtxt(file_path+filename, delimiter=',')
    data = raw_signal[5*256:6*256, electrodes]
    return prepareData(data.T)

def signal_pop(choice):
    data = choose_data(choice)
    signal = np.repeat(data, targets/4, axis=0)
    return signal

#random population
def randpop():
    random_pop = np.round(np.random.rand(targets,256), 2)
    return random_pop
#sinusoid population
t=np.linspace(0,1,256)
sin14 = np.full((int(targets/2),256), np.sin(2*np.pi*t*14))
sin8 = np.full((targets,256), np.sin(2*np.pi*t*8))
cos14 = np.full((int(targets/2),256), np.cos(2*np.pi*t*14))
cos8 = np.full((int(targets/2),256), np.cos(2*np.pi*t*8))
def popsin(sin,cos):
    pop1 = np.round(normalize(sin), 2)
    pop2 = np.round(normalize(cos), 2)
    pop = (np.concatenate((pop1,pop2), axis=0))
    np.random.shuffle(pop)
    return pop

popsin14 = popsin(sin14, cos14)
popsin8 = popsin(sin8, cos8)

#plt.plot(random_pop[random.randint(0,(len(random_pop)-1))])
#plt.show()
