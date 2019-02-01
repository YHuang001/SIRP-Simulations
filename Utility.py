# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:43:13 2019

@author: yufan
"""

import networkx as nx
import numpy as np
import math
from scipy import spatial
from collections import defaultdict


def pSample(rate):
    """
    Generate a Poisson distributed random value
    """
    return -np.log(np.random.random())/rate

def times(pairs1, pairs2, rate1, rate2):
    """
    Generate Poisson distributed times for each edge
    """
    t1, t2 = defaultdict(float), defaultdict(float)

    for pair1, pair2 in zip(pairs1, pairs2):
        t1[pair1] = pSample(rate1)
        t2[pair2] = pSample(rate2)
    
    return t1, t2

def shiftE(pairs, nodes, nC):
    """
    Shift the edges' labels according to the cluster number
    """
    return [(e1 + nodes[nC][0], e2 + nodes[nC][0]) for e1, e2 in pairs]

def tER(n, AVE1, AVE2, sim):
    """
    Generate node pairs according to two-ER networks
    """
    p1, p2 = AVE1/n, AVE2/n
    G1 = nx.fast_gnp_random_graph(n, p1 - np.sqrt(p1*p2)*sim)
    G2 = nx.fast_gnp_random_graph(n, p2 - np.sqrt(p1*p2)*sim)
    Gm = nx.fast_gnp_random_graph(n, np.sqrt(p1*p2)*sim)
    mpairs = set(Gm.edges())
    pairs1 = list(mpairs.union(set(G1.edges())))
    pairs2 = list(mpairs.union(set(G2.edges())))
    
    return pairs1, pairs2

def tRGG(n, AVE1, AVE2):
    """
    Generate node pairs according to two-RGG networks
    """
    positions = np.random.rand(n, 2)
    kdtree = spatial.KDTree(positions)
    r1, r2 = np.sqrt(AVE1/n/math.pi), np.sqrt(AVE2/n/math.pi)
    pairs1, pairs2 = list(kdtree.query_pairs(r1)), list(kdtree.query_pairs(r2))
    
    return pairs1, pairs2

def ConsInts(topo, N, nlinks, p):
    """
    Generate interconnections for the corresponding cluster structure
    """
    TInters = {}
    if topo == 'line':
        num = list(range(N))
        for pair in tuple(zip(num[:-1], num[1:])):
            TInters[pair] = nlinks
    elif topo == 'star':
        num = list(range(1, N))
        for pair in tuple(zip([0]*(N-1), num)):
            TInters[pair] = nlinks        
    elif topo == 'random':
        G = nx.fast_gnp_random_graph(N, p)
        for pair in G.edges:
            TInters[pair] = nlinks
    else:
        raise ValueError("Topology not supported")
    return TInters

def MultiCluster(TSizes, TTypes, TParams, rate1, rate2,
                 Depend=False, TIs1={}, TIs2={}):
    """
    Generate the multi-cluster graphs
    """
    G1, G2 = nx.Graph(), nx.Graph()
    
    nodes = []
    for size in TSizes:
        if not nodes:
            node = np.arange(0, size)
            nodes.append(node)
            G1.add_nodes_from(node)
            G2.add_nodes_from(node)
        else:
            node = np.arange(nodes[-1][-1] + 1, nodes[-1][-1] + size + 1)
            nodes.append(node)
            G1.add_nodes_from(node)
            G2.add_nodes_from(node)
            
    for i, size in enumerate(TSizes):
        if TTypes[i] == 'ER':
            pairs1, pairs2 = tER(size, TParams[i][0], 
                                 TParams[i][1], TParams[i][2])
            pairs1 = shiftE(pairs1, nodes, i)
            pairs2 = shiftE(pairs2, nodes, i)
            G1.add_edges_from(pairs1)
            G2.add_edges_from(pairs2)
        elif TTypes[i] == 'GR':
            pairs1, pairs2 = tRGG(size, TParams[i][0], TParams[i][1])
            pairs1 = shiftE(pairs1, nodes, i)
            pairs2 = shiftE(pairs2, nodes, i)
            G1.add_edges_from(pairs1)
            G2.add_edges_from(pairs2)
        else:
            raise TypeError('Graph type does not exist!')
            
    if len(TSizes) > 1:
        inter_pairs1, inter_pairs2 = set(), set()
        if not Depend:
            for key, value in TIs1.items():
                for i in range(value):
                    node1 = np.random.choice(nodes[key[0]])
                    node2 = np.random.choice(nodes[key[1]])
                    while (node1, node2) in inter_pairs1:
                        node2 = np.random.choice(nodes[key[1]])
                    inter_pairs1.add((node1, node2))
            for key, value in TIs2.items():
                for i in range(value):
                    node1 = np.random.choice(nodes[key[0]])
                    node2 = np.random.choice(nodes[key[1]])
                    while (node1, node2) in inter_pairs2:
                        node2 = np.random.choice(nodes[key[1]])
                    inter_pairs2.add((node1, node2))         
        else:
            for key, value in TIs1.items():
                cnt = 0
                if TIs2[key] <= value:
                    for i in range(value):
                        node1 = np.random.choice(nodes[key[0]])
                        node2 = np.random.choice(nodes[key[1]])
                        while (node1, node2) in inter_pairs1:
                            node2 = np.random.choice(nodes[key[1]])
                        inter_pairs1.add((node1, node2))
                        if cnt < TIs2[key]:
                            inter_pairs2.add((node1, node2))
                            cnt += 1
                else:
                    for i in range(TIs2[key]):
                        node1 = np.random.choice(nodes[key[0]])
                        node2 = np.random.choice(nodes[key[1]])
                        while (node1, node2) in inter_pairs2:
                            node2 = np.random.choice(nodes[key[1]])
                        inter_pairs2.add((node1, node2))
                        if cnt < value:
                            inter_pairs1.add((node1, node2))
                            cnt += 1
        G1.add_edges_from(inter_pairs1)
        G2.add_edges_from(inter_pairs2)
    T1, T2 = times(list(G1.edges()), list(G2.edges()), rate1, rate2)

    Neigh1 = [set(G1.neighbors(i)) for i in range(nodes[-1][-1] + 1)]
    Neigh2 = [set(G2.neighbors(i)) for i in range(nodes[-1][-1] + 1)]
    
    return T1, T2, Neigh1, Neigh2
                