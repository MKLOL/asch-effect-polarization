"""
File to test the intersection between stooges if we try to maximize / minimze.
"""

from graph_creator import getGraph
from mse_stooges_resistance_greedy import *
import random
import numpy as np
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
        ret[x] = (len(ls), sum( (b[a] - avg) * (b[a] - avg) for a in ls))
    return ret

def makeData(lsPos, lsNeg):
    print("best full greedy ratio:", lsPos[-1] / lsPos[0])
    ls1 = []
    for i in range(len(lsPos)):
        ls1.append(lsPos[i] / lsPos[0])
        print(lsPos[i] / lsPos[0], i)

    print("best random greedy ratio:", lsNeg[-1] / lsNeg[0])

    ls2 = []
    for i in range(len(lsNeg)):
        ls2.append(lsNeg[i] / lsNeg[0])
        print(lsNeg[i] / lsNeg[0], i)
    return lsNeg[-1]/lsNeg[0], lsPos[-1] / lsPos[0]

def getRandom(G, s, stoogeCount):
    nodes = list(G.nodes)
    random.shuffle(nodes)
    nodes = nodes[:stoogeCount]
    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False)
    stoogeNeg, resistNeg, lsNeg = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False, change_nodes=nodes)
    return makeData(lsPos, lsNeg)

def getRandomCentrality(G, s, stoogeCount):
    ls = sorted([(len(list(G.neighbors(x))), x) for x in G.nodes])[::-1]
    nodes = [x[1] for x in ls]
    nodes = nodes[:stoogeCount]
    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False)
    stoogeNeg, resistNeg, lsNeg = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False, change_nodes=nodes)
    return makeData(lsPos, lsNeg)


s1 = []
s2 = []
for i in range(10):
    G, s = getGraph("star_random")
    l1, l2 = getRandomCentrality(G, s, int(math.log2(len(G.nodes)) * 5))
    s1.append(l1)
    s2.append(l2)

print(np.average(s1), np.average(s2), np.std(s1), np.std(s2))
