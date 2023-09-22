import numpy as np
from mse_graph_calculator import *


def getStoogeList(G, stoogeCount):
    n = len(G.nodes)
    s = np.clip(np.random.normal(0.5, 0.5, n),0,1)
    stoogeDict = dict()
    start = 0
    for i in range(stoogeCount):
        a, b = calculateMse(G, s, 1)
        if start == 0:
            start = b
        best = (0, 0, 0, 1.0)
        for ns in range(n):
            if stoogeDict.get(ns) is not None:
                continue
            sn = list(s)
            sn[ns] = 1.0
            init, comp = calculateMse(G, sn, 1)
            if best[0] < (comp - b):
                best = (comp - b, ns, 1.0, comp)
            sn[ns] = 0.0
            init, comp = calculateMse(G, sn, 1)
            if best[0] < (comp - b):
                best = (comp - b, ns, 0.0, comp)
        print(best, b, best[3]/max(0.00001, start))
        if best[0] != 0:
            s[best[1]] = best[2]
            stoogeDict[best[1]] = best[2]
    return (stoogeDict,s)
