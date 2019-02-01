# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:12:39 2019

@author: yufan
"""

import heapq
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Utility import MultiCluster, ConsInts
from collections import defaultdict
from math import ceil


N = 50
SC = 200
GAMMA, REC = 0.3, 1.0
SEC = 5

def addIevent(e, n, s, t):
    """
    Push the infection event into the heap
    """
    heapq.heappush(e, (t, (n, s, 'I')))

def addPevent(e, n, s, t):
    """
    Push the protection event into the heap
    """
    heapq.heappush(e, (t, (n, s, 'P')))

def sStatus(node, st):
    """
    Determine if a node is susceptible
    """
    if node not in st['R'] and node not in st['I'] and node not in st['P']:
        return True
    return False
    
def simulation(n, t1, t2, n1, n2, startI, columns, beta, npls, Monte = 100):
    """
    SIPR process simulation
    """ 
    print(startI)
    index = range(startI, startI + Monte)
    df = pd.DataFrame(index = index, columns = columns)
    for run in index:
        status = defaultdict(set)
        Icluster = [0]*N
        source = int(n*np.random.rand())
        Icluster[int(source/SC)] = 1
        status['I'].add(source)
        events = []
        heapq.heapify(events)
        for neigh in n1[source]:
            if (neigh, source) in t1:
                addIevent(events, neigh, source, t1[(neigh, source)])
            elif (source, neigh) in t1:
                addIevent(events, neigh, source, t1[(source, neigh)])
        for neigh in n2[source]:
            if (neigh, source) in t2:
                addPevent(events, neigh, source, t2[(neigh, source)])
            elif (source, neigh) in t2:
                addPevent(events, neigh, source, t2[(source, neigh)])  
        heapq.heappush(events, (REC, (source, source, 'R')))
        while len(events):
            t, new_event = heapq.heappop(events)
            new_end, old_source, eventType = new_event
            if eventType == 'R':
                status['R'].add(old_source)
            elif eventType == 'P':
                if old_source not in status['R']:
                    if sStatus(new_end, status):
                        status['P'].add(new_end)
            elif eventType == 'I':
                if old_source not in status['R']:
                    if sStatus(new_end, status):
                           status['I'].add(new_end)
                           clusterN = int(new_end/SC)
                           if not Icluster[clusterN]:
                               Icluster[clusterN] = 1
                           for neigh in n1[new_end]:
                               if sStatus(neigh, status):
                                   if (neigh, new_end) in t1:
                                       addIevent(events, neigh, new_end, 
                                                 t + t1[(neigh, new_end)])
                                   elif (new_end, neigh) in t1:
                                       addIevent(events, neigh, new_end, 
                                                 t + t1[(new_end, neigh)])
                           for neigh in n2[new_end]:
                               if sStatus(neigh, status):
                                   if (neigh, new_end) in t2:
                                       addPevent(events, neigh, new_end, 
                                                 t + t2[(neigh, new_end)])
                                   elif (new_end, neigh) in t2:
                                       addPevent(events, neigh, new_end, 
                                                 t + t2[(new_end, neigh)])
                           heapq.heappush(events, (t + REC, 
                                                   (new_end, new_end, 'R')))
                           
        df.loc[run] = pd.Series({'R0': beta*10,
                                 'R_POR': (len(status['R'])/n), 
                                 'P_POR': (len(status['P'])/n),
                                 'ClusterS': ceil(sum(Icluster)/SEC)*SEC,
                                 'NumClusters': sum(Icluster)})  
 
    return df

if __name__ == '__main__':
    start = time.time()
    Sizes = [SC]*N
    Types = ['GR']*N
    TParams = [[10, 10, 1]]*N
    nlinks = 5
    TInters1 = ConsInts('line', N, nlinks, 5/N)
    TInters2 = {}
    BVs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    columns = ['R0', 'R_POR', 'P_POR', 'ClusterS', 'NumClusters']
    ResL, startI, npls = [], 0, 10
    for beta in BVs:
        rate1, rate2 = -np.log(1 - beta)/REC, -np.log(1 - GAMMA)/REC
        Nrep, Monte = 10, 100
        for rep in range(Nrep):
            t1, t2, n1, n2  = MultiCluster(Sizes, Types, TParams, rate1, rate2,
                                           Depend = False, 
                                           TIs1 = TInters1, TIs2 = TInters2)    
            ResL.append(simulation(sum(Sizes), t1, t2, n1, n2, startI, 
                                   columns, beta, npls, Monte = Monte))
            startI += Monte
    Res = pd.concat(ResL)
    plt.figure(3)
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['legend.markerscale'] = 2
    ax = sns.scatterplot(x = 'R0', y = 'R_POR', hue = 'ClusterS', 
                         data = Res, 
                         palette = sns.color_palette("coolwarm", 
                                    len(Res['ClusterS'].unique())), s = 200)
    end = time.time()
    print(end - start)