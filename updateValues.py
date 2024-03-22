import random

import networkx
import matplotlib
import graph_construction
from mse_stooges_resistance_greedy import *
import random
import numpy as np

probLimit = 1e-10
updateCount = 100000


def rw(cur_node, cur_prob, path, retOpinions, G, initialOpinions, stableOpinions, resistances, rootNode, cache):
    if cur_prob < probLimit:
        return
    if len(path) > 0 and cur_node == rootNode:
        return
    prob_edge = 1 - resistances[cur_node]
    if len(path) > 0:
        ns = G.neighbors(cur_node)
        cur_prob *= prob_edge * (1 / (len(list(ns))))
    nextNode = random.choice(list(G.neighbors(cur_node)))
    if tuple(path) not in cache and len(path) > 0:
        retOpinions[cur_node] -= stableOpinions[rootNode] * cur_prob
        retOpinions[cur_node] += initialOpinions[rootNode] * cur_prob

    cache.add(tuple(path))
    path.append(cur_node)
    rw(nextNode, cur_prob, path, retOpinions, G, initialOpinions, stableOpinions, resistances, rootNode, cache)


def update(G, initialOpinions, stableOpinions, resistances, node):
    ret = list(stableOpinions)
    cache = set()
    for x in range(updateCount):
        rw(node, 1.0, [], ret, G, initialOpinions, stableOpinions, resistances, node, cache)
    ret[node] = initialOpinions[node]
    return ret


G = networkx.erdos_renyi_graph(320, 0.2)
n = len(G.nodes)
s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
resistances = [0.5] * n
a, b = approximateMseFaster(G, s, resistances=resistances)
resistances2 = list(resistances)
resistances2[5] = 1.0
mse1, x = approximateMseFaster(G, s, resistances=resistances2)

print(x)
y = update(G, s, b, resistances, 5)

print(y)

for i in range(len(x)):
    print(x[i] - y[i])
print()
