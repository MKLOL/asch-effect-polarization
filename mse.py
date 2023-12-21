import random

import networkx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import graph_construction
from mse_stooges_resistance_greedy import *

"""
n = 44

G = nx.erdos_renyi_graph(n, p=0.3)
G.add_edges_from(zip(G.nodes, G.nodes))
edgeCount = 10

print(getEdgeList(G, edgeCount))"""
# G,s = graph_construction.makeGraphFromFile("edge-twoclusters.in")
# G = networkx.erdos_renyi_graph(320, 0.2)
G = networkx.star_graph(320)
n = len(G.nodes)
s = np.clip(np.random.normal(0.5, 0.5, n),0,1)
G.add_edges_from(zip(G.nodes, G.nodes))

s = []
s.append(1)
for x in range(n-1):
    s.append(0.498433)

a, b = approximateMseFaster(G, s)
y = [x * x for x in b]
mn = np.mean(b)
print(mn)
z = [abs(x - mn) for x in b]
plt.hist(b, bins=20, color='blue', edgecolor='black')
plt.show()
plt.hist(y, bins=20, color='blue', edgecolor='black')
plt.show()
plt.hist(z, bins=20, color='blue', edgecolor='black')
plt.show()
ls, resist = greedyResistance(G, s, 12)
print(len(G.nodes))
print("best ratio:", ls[-1]/ls[0])
for i in range(len(ls)):
    print(ls[i]/ls[0], i)


a, b = approximateMseFaster(G, s, resistances=resist)
y = [x *x for x in b]
mn = np.mean(b)
print(mn)
z = [abs(x - mn) for x in b]

plt.hist(b, bins=20, color='blue', edgecolor='black')
plt.show()
plt.hist(y, bins=20, color='blue', edgecolor='black')
plt.show()
plt.hist(z, bins=20, color='blue', edgecolor='black')
plt.show()


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


star init and star final run on 320 star graph node. 
Final ration 32. 
"""