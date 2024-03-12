import numpy as np
import networkx as nx
from mse_graph_calculator import *


def greedyResistanceMega(G, stoogeCount, baseResistance = 0.5):
    n = len(G.nodes)
    resistances = baseResistance * np.ones(n)

    s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
    stoogeDict = {}

    for i in range(stoogeCount):
        _, mse0 = calculateMse(G, s, resistances=resistances)
        #print(f"Iteration {i}: MSE={mse0}")

        mse_max = mse0
        x_max = None
        r_max = None
        op_max = None
        for x in range(n):
            if x in stoogeDict: continue
            for r in [0, 1]:
                for op in [0, 1]:
                    slist = list(s)
                    slist[x] = op
                    resistances1 = np.copy(resistances)
                    resistances1[x] = r

                    _, mse1 = calculateMse(G, slist, resistances=resistances1)

                    if mse1 > mse_max:
                        mse_max = mse1
                        x_max = x
                        r_max = r
                        op_max = op

        if x_max is None: break

        resistances[x_max] = r_max
        stoogeDict[x_max] = True
        s[x_max] = op_max
        #print(f"  resistance({x_max})={r_max} (MSE={mse_max}) opinion({x_max})={op_max})")

    return resistances, mse_max

