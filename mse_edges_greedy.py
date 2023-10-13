import numpy as np
from mse_graph_calculator import *


def getEdgeList(G, edgeCount):
    n = len(G.nodes)
    s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
    all_edges = set(G.edges)
    ret = []

    a, b = calculateMse(G, s, 1)
    print(a,b)
    for i in range(edgeCount):
        a, b = calculateMse(G, s, 1)
        print(a, b)
        best = (b, None)
        for i in range(len(G.nodes)):
            for j in range(len(G.nodes)):
                if i == j:
                    continue
                if (i, j) in all_edges or (j, i) in all_edges:
                    continue

                G.add_edge(i, j)
                a, b = calculateMse(G, s, 1)
                G.remove_edge(i, j)
                if b > best[0]:
                    best = (a, (i, j))
        for i in all_edges:
            if i[0] == i[1]:
                continue
            #print(i, all_edges)
            #print(i, G.edges)
            G.remove_edge(i[0], i[1])
            a, b = calculateMse(G, s, 1)
            G.add_edge(i[0], i[1])
            if b > best[0]:
                best = (a, (-i, -j))
        if best[1][1] != None:
            if best[1][1] < 0:
                G.remove_edge(best[1][0], best[1][1])
                all_edges.remove(best[1])
                ret.append(("-", best[1]))
            else:
                G.add_edge(best[1][0], best[1][1])
                all_edges.add(best[1])
                ret.append(("+", best[1]))

    return ret
