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

G, s = getGraph("STAR")


def getRandom(G, s, stoogeCount):
    nodes = list(G.nodes)
    random.shuffle(nodes)
    nodes = nodes[:stoogeCount]
    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False)
    stoogeNeg, resistNeg, lsNeg = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False, change_nodes=nodes)
    print("best full greedy ratio:", lsPos[-1] / lsPos[0])

    for i in range(len(lsPos)):
        print(lsPos[i] / lsPos[0], i)

    print("best random greedy ratio:", lsNeg[-1] / lsNeg[0])

    for i in range(len(lsNeg)):
        print(lsNeg[i] / lsNeg[0], i)

n = len(G.nodes)
target = None
G.add_edges_from(zip(G.nodes, G.nodes))
if type == "GNP" or type == "star_random" or type == "tree":
    s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)


if type == "STAR":
    s.append(1)
    for x in range(n - 1):
        s.append(0.498433)
if type == "d_regular":
    s = np.clip(np.random.normal(0.8, 0.5, n // 2), 0, 1)
    y = np.clip(np.random.normal(0.2, 0.5, n // 2), 0, 1)
    s = np.append(s, y)
    print(len(s), len(y))

getRandom(G, s, int(math.log2(len(G.nodes)) * 5))