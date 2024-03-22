from utils import getGraph
from mse_stooges_resistance_greedy import *
import math

"""
Takes a graph, initial opinions and stoogeCount and simulates the greedy algorithm
Then for every nodes it computes the distances to closes stooge and returns for every distance average MSE.
"""


def getDistGraph(graph, initOpinions, stoogeCount):
    stoogeInformation, resistPos, mse0s = greedyResistanceNegative(graph, initOpinions, stoogeCount, positive=True)
    stoogeNodes = set([stoogeInfo[1] for stoogeInfo in stoogeInformation])
    layers = nx.bfs_layers(graph, stoogeNodes)
    ret = dict()
    a, b = approximateMseFaster(graph, initOpinions, resistances=resistPos)
    avg = sum(b) / len(b)
    hd = dict(enumerate(layers))
    for node in hd:
        distance = hd[node]
        ret[node] = (len(distance), sum((b[a] - avg) * (b[a] - avg) for a in distance))
    return ret


aggregateComputation = dict()
expCount = 5
for tt in range(expCount):
    G, s = getGraph("GNP")
    n = len(G.nodes)
    r = getDistGraph(G, s, int(math.log2(len(G.nodes)) * 5))
    for x in r:
        if aggregateComputation.get(x) is None:
            aggregateComputation[x] = (0, 0)
        aggregateComputation[x] = (aggregateComputation[x][0] + r[x][0], aggregateComputation[x][1] + r[x][1])

averagePerDist = [0] * len(aggregateComputation)

for x in aggregateComputation:
    averagePerDist[x] = float(aggregateComputation[x][1] / aggregateComputation[x][0])

plt.plot(averagePerDist)
plt.show()
