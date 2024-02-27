"""
File to test the intersection between stooges if we try to maximize / minimze.
"""

import matplotlib.pyplot as plt

from graph_creator import getGraph
from mse_stooges_resistance_greedy import *
import math


def getDistGraph(G, s, stoogeCount):
    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, stoogeCount, positive=True)
    sps = set([x[1] for x in stoogePos])
    layers = nx.bfs_layers(G, sps)
    ret = dict()
    a, b = approximateMseFaster(G, s, resistances=resistPos)
    avg = sum(b) / len(b)
    hd = dict(enumerate(layers))
    for x in hd:
        print(x)
        ls = hd[x]
        print(len(ls))
        ret[x] = (len(ls), sum((b[a] - avg) * (b[a] - avg) for a in ls))
    return ret


"""
n = 44

G = nx.erdos_renyi_graph(n, p=0.3)
G.add_edges_from(zip(G.nodes, G.nodes))
edgeCount = 10

print(getEdgeList(G, edgeCount))"""
allRet = dict()
expCount = 5
for tt in range(expCount):
    G, s = getGraph("GNP")
    n = len(G.nodes)
    r = getDistGraph(G, s, int(math.log2(len(G.nodes)) * 5))
    for x in r:
        if allRet.get(x) == None:
            allRet[x] = (0, 0)
        allRet[x] = (allRet[x][0] + r[x][0], allRet[x][1] + r[x][1])

ls = [0] * len(allRet)

for x in allRet:
    ls[x] = float(allRet[x][1] / allRet[x][0])
print(ls)
plt.plot(ls)
plt.show()
