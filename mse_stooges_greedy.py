import numpy as np
from mse_graph_calculator import *


def getStoogeList(G, stoogeCount):
    n = len(G.nodes)
    s = np.random.normal(0, 1, n)
    stoogeDict = dict()
    for i in range(stoogeCount):
        a, b = calculateMse(G, s, 1)
        best = (0, 0, 0)
        for ns in range(n):
            if stoogeDict.get(ns) is not None:
                continue
            sn = list(s)
            sn[ns] = 1.0
            init, comp = calculateMse(G, sn, 1)
            if best[0] < (comp - b):
                best = (comp - b, ns, 1.0)
            sn[ns] = 0.0
            init, comp = calculateMse(G, sn, 1)
            if best[0] < (comp - b):
                best = (comp - b, ns, 0.0)
        if best[0] != 0:
            s[best[1]] = best[2]
            stoogeDict[best[1]] = best[2]
    return (stoogeDict,s)
