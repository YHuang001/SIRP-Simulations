# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:47:33 2019

@author: yufan
"""

import numpy as np
import heapq
from Utility import MultiCluster, ConsInts
from collections import defaultdict
import time


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
    
def simulation(n, t1, t2, n1, n2, Monte = 100):
    """
    SIPR process simulation
    """
    R_por, P_por = 0, 0
    for run in range(Monte):
        print(run)
        status = defaultdict(set)
        source = int(n*np.random.rand())
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
        R_por += len(status['R'])/n
        P_por += len(status['P'])/n
    R_por /= Monte
    P_por /= Monte
    return R_por, P_por

if __name__ == '__main__':
    start = time.time()
    N = 1
    Sizes = [100]*N
    Types = ['GR']*N
    TParams = [[10, 10, 1]]*N
    nlinks = 3
    TInters1 = ConsInts('line', N, nlinks, 5/N)
    TInters2 = {}
    BETA, GAMMA, REC = 0.5, 0.3, 1.0
    rate1, rate2 = -np.log(1 - BETA)/REC, -np.log(1 - GAMMA)/REC
    t1, t2, n1, n2  = MultiCluster(Sizes, Types, TParams, rate1, rate2,
                                   Depend = False, 
                                   TIs1 = TInters1, TIs2 = TInters2)    
    R_por, P_por = simulation(sum(Sizes), t1, t2, n1, n2)
    end = time.time()
    print(R_por, P_por, end - start)