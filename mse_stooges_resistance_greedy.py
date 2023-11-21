import numpy as np
import networkx as nx
from mse_graph_calculator import *


def greedyResistance(G, initialOpinions, stoogeCount, baseResistance=0.5, change_nodes=None, targetNodes = None, verbose=True):
    n = len(G.nodes)
    resistances = baseResistance * np.ones(n)
    targetNodeSet = set()
    if targetNodes is not None:
        targetNodeSet = set(targetNodes)
    # s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
    stoogeDict = {}
    change_nodes = G.nodes if change_nodes is None else change_nodes

    x_start = None
    active_nodes = G.nodes

    mse0, x_start = approximateMseFaster(G, initialOpinions, resistances=resistances, x_start=x_start, active_nodes=active_nodes, targetNodes=targetNodes)
    mse0s = [mse0]

    for i in range(stoogeCount):
        mse_max = mse0
        x_max = None
        r_max = None

        for x in change_nodes:
            if x in stoogeDict: continue
            if x in targetNodeSet: continue
            for r in [0, 1]:
                resistances1 = np.copy(resistances)
                resistances1[x] = r

                mse1, _ = approximateMseFaster(G, initialOpinions, resistances=resistances1, x_start=x_start, active_nodes=[x], targetNodes=targetNodes)

                if mse1 > mse_max:
                    mse_max = mse1
                    x_max = x
                    r_max = r

            if verbose: print(".", end="", flush=True)
            # if (n - x) % 10 == 0: print(f"    {n - x} remaining")

        if x_max is None:
            print("x_max is none, thus breaking")
            break

        resistances[x_max] = r_max
        stoogeDict[x_max] = True

        mse0, x_start = approximateMseFaster(G, initialOpinions, resistances=resistances, x_start=x_start, active_nodes=active_nodes, targetNodes=targetNodes)
        mse0s.append(mse0)
        if verbose: print(f"\nIteration {i}: MSE={mse0} (setting resistance({x_max})={r_max})")

    return mse0s # resistances, mse_max

