# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 20:31:47 2019

@author: yufan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:47:33 2019

@author: yufan
"""


import heapq
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
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
    
def simulation(n, t1, t2, n1, n2):
    """
    SIPR process simulation and generate genealogy for the infected nodes
    """
    status = defaultdict(set)
    generation = defaultdict(int)
    infG = nx.DiGraph()
    index = range(n)
    columns = ['Id', 'Parent', 'Generation']
    records = pd.DataFrame(index = index, columns = columns)
    source = 0
    color_map = []
    posxR, posyR, currentL, currentG = 1/n, 3/n, 1, 1
    infG.add_node(source, pos=(posxR*currentL, posyR*currentG))
    cmap = matplotlib.cm.get_cmap('tab10')
    color_map.append(cmap(source/1000))
    status['I'].add(source)
    generation[source] = 1
    cnt = 0
    records.loc[cnt] = pd.Series({'Id': source, 'Parent': np.nan, 
                                 'Generation': generation[source]})
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
                       cnt += 1
                       generation[new_end] = generation[old_source] + 1
                       records.loc[cnt] = pd.Series({'Id': new_end, 
                                        'Parent': old_source, 
                                        'Generation': generation[new_end]})
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
                       
    records = records.sort_values(by = ['Generation'])
    for index, row in records.iterrows():
        if not pd.isna(row['Parent']):
            if row['Generation'] != currentG:
                currentL = 1
                currentG = row['Generation']
            else:
                currentL += 1
            infG.add_node(row['Id'], pos=(posxR*currentL, posyR*currentG))
            infG.add_edge(row['Parent'], row['Id'])
            color_map.append(cmap(row['Id']/1000))
    options = {
        'node_size': 300,
        'width': 1,
        'arrowstyle': '-|>',
        'arrowsize': 10,
    }
    pos=nx.get_node_attributes(infG,'pos')
    nx.draw_networkx(infG, node_color = color_map, 
                     pos=pos, arrows=True, **options, font_size = 6)
        

if __name__ == '__main__':
    start = time.time()
    N = 2
    Sizes = [100]*N
    Types = ['ER']*N
    TParams = [[10, 10, 1]]*N
    nlinks = 4
    TInters1 = ConsInts('line', N, nlinks, 2/N)
    TInters2 = {}
    BETA, GAMMA, REC = 0.8, 0.3, 1.0
    rate1, rate2 = -np.log(1 - BETA)/REC, -np.log(1 - GAMMA)/REC
    t1, t2, n1, n2  = MultiCluster(Sizes, Types, TParams, rate1, rate2,
                                   Depend = False, 
                                   TIs1 = TInters1, TIs2 = TInters2)    
    simulation(sum(Sizes), t1, t2, n1, n2)
    end = time.time()
    print(end - start)