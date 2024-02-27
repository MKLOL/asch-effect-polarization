import networkx
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
import graph_construction
from graph_creator import getGraph, write_statistics_to_file
from mse_stooges_resistance_greedy import *
import random
import numpy as np
import math
from networkx.algorithms import bipartite


"""
n = 44

G = nx.erdos_renyi_graph(n, p=0.3)
G.add_edges_from(zip(G.nodes, G.nodes))
edgeCount = 10

print(getEdgeList(G, edgeCount))"""
seed = 123
random.seed(seed)
np.random.seed(seed)
"""
Main types are: GNP, START, and anything else

Anything else signifies you're reading from an .in file. 

d_regular, tree, STAR
"""
type = "d_regular"
status = "pre"

G, s = getGraph(type)

n = len(G.nodes)
target = None
positive = False

a, b = approximateMseFaster(G, s)
y = [x * x for x in b]
binit = b
mn = np.mean(b)

z = [abs(x - mn) for x in b]
write_statistics_to_file(f"{status}_{seed}_{type}_stats.txt", b)
plt.title(f"SEED {seed}, Plotting opinions, {status} greedy. {type}")
plt.hist(b, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_opinion.png')

plt.title(f"SEED {seed}, Plotting x*x, {status} greedy {type}")
plt.hist(y, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_square.png')

plt.title(f"SEED {seed}, Plotting abs(x-avg), {status} greedy {type}")
plt.hist(z, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_abs.png')

_, resist, ls = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=positive)
print(len(G.nodes))
print("best ratio:", ls[-1] / ls[0])
lx = [i / ls[0] for i in ls]
print(ls)
print(lx)
plt.title(f"SEED {seed}, Plotting Stooges, {status} greedy {type}")

plt.cla()
plt.clf()
plt.plot(lx, color='blue')
plt.savefig(f'{status}_{seed}_{type}_stooge.png')
for i in range(len(ls)):
    print(ls[i] / ls[0], i)

plt.title(f"SEED {seed}, Plotting stooges vs MSE: {status} greedy {type}")
plt.hist(ls, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_mse_by_time.png')

a, b = approximateMseFaster(G, s, resistances=resist, targetNodes=target)
y = [x * x for x in b]
mn = np.mean(b)
z = [abs(x - mn) for x in b]

bfin = b


z2 = [abs(bfin[x] - binit[x]) for x in range(len(bfin))]

status = "post"
write_statistics_to_file(f"{status}_{seed}_{type}_stats.txt", b)
plt.cla()
plt.clf()
plt.title(f"SEED {seed}, Plotting opinions, {status} greedy. {type}")
plt.hist(b, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_opinion.png')

plt.title(f"SEED {seed}, Plotting x*x, {status} greedy {type}")
plt.hist(y, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_square.png')

plt.title(f"SEED {seed}, Plotting abs(x-avg), {status} greedy {type}")
plt.hist(z, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_abs.png')

plt.cla()
plt.clf()

plt.title(f"SEED {seed}, Plotting abs(initStable-finalStable), {status} greedy {type}")
plt.hist(z2, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_abs_init_final.png')

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
