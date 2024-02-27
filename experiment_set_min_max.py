"""
File to test the intersection between stooges if we try to maximize / minimze.
"""

from graph_creator import getGraph
from mse_stooges_resistance_greedy import *
import numpy as np
import math





"""
n = 44

G = nx.erdos_renyi_graph(n, p=0.3)
G.add_edges_from(zip(G.nodes, G.nodes))
edgeCount = 10

print(getEdgeList(G, edgeCount))"""


type = "smallCommunities"
status = "pre"

G, s = getGraph(type)
n = len(G.nodes)
target = None

a, b = approximateMseFaster(G, s)
y = [x * x for x in b]
binit = b
mn = np.mean(b)

z = [abs(x - mn) for x in b]

stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=True)
stoogeNeg, resistNeg, lsNeg = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False)
print("best positive ratio:", lsPos[-1] / lsPos[0])

for i in range(len(lsPos)):
    print(lsPos[i] / lsPos[0], i)

print("best positive ratio:", lsNeg[-1] / lsNeg[0])

for i in range(len(lsNeg)):
    print(lsNeg[i] / lsNeg[0], i)

sps = set([x[1] for x in stoogePos])
sns = set([x[1] for x in stoogeNeg])

print(sps, sns)
print(len(set(stoogeNeg)), len(set(stoogePos)), len(sps.intersection(sns)), sps.intersection(sns))