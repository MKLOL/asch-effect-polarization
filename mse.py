import random

import graph_construction
from mse_stooges_greedy import *
from mse_graph_calculator import *
from mse_edges_greedy import *
from mse_stooges_resistance_greedy import *
from mse_stooges_mega_greeedy import *
import networkx as nx
"""
n = 44

G = nx.erdos_renyi_graph(n, p=0.3)
G.add_edges_from(zip(G.nodes, G.nodes))
edgeCount = 10

print(getEdgeList(G, edgeCount))"""
n = 150
G = graph_construction.makeGraphFromFile("edge-twoclusters.in")
#G = nx.erdos_renyi_graph(n, p=0.3)
print(G)
n = len(G.nodes)
G.add_edges_from(zip(G.nodes, G.nodes))
targetNodes = random.sample(range(len(G.nodes)//2), k=40)
print(targetNodes)
s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
ls = greedyResistance(G, s, 60, targetNodes=targetNodes)
print(len(G.nodes))
print(ls[-1]/ls[0])
for i in range(len(ls)):
    print(ls[i]/ls[0], i)

print(ls)
"""
130
10,20,30 nodes
15,30,33 % increase

150
10,20,30
16,29,42

230 (160 max)
10,20,30,40,60
7,18,21,23,32


TWO CLUSTERS:
320 (160 - 160)
1.07126948861406 10
1.1394600994370023 20
1.2074788354896184 30
1.2752423318981125 40
1.3424463722417608 50
1.4004436419655881 59

"""