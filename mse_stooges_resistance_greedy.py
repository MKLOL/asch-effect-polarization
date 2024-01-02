import numpy as np
import networkx as nx
from mse_graph_calculator import *


"""
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

    return mse0s, resistances # resistances, mse_max
"""


def lazy_greedy(f, xs, k):
    n = len(xs)
    picked = []
    current_val = f(picked)

    marginal_gains = np.inf * np.ones(n)
    for i in range(k):
        best_j = None
        print(f"{i}: f({', '.join(map(str, picked))})={current_val}")
        prev_gain = 0
        for j in np.argsort(marginal_gains)[::-1]:
            if xs[j] in picked: continue
            if marginal_gains[j] <= prev_gain: break
            gain = f(picked + [xs[j]]) - current_val
            marginal_gains[j] = gain
            if best_j is None or gain > marginal_gains[best_j]: best_j = j
            print(".", end="", flush=True)
            prev_gain = gain

        print("")
        if best_j is None:
            print("no further improvement possible")
            break
        picked.append(xs[best_j])
        current_val += marginal_gains[best_j]

    return picked



def greedyResistance(G, initialOpinions, stoogeCount, baseResistance=0.5, change_nodes=None, targetNodes = None, verbose=True):
    n = len(G.nodes)
    resistances = baseResistance * np.ones(n)
    targetNodeSet = set()
    if targetNodes is not None:
        targetNodeSet = set(targetNodes)
    # s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
    change_nodes = G.nodes if change_nodes is None else change_nodes

    current_x_start = {}

    def calc_mse(stooges):
        resistances1 = np.copy(resistances)
        for x, r in stooges:
            resistances1[x] = r

        x_start = current_x_start.get(stooges[-2] if len(stooges) > 1 else None, None)
        active_nodes = [stooges[-1][0]] if len(stooges) > 1 else None

        mse, x_start = approximateMseFaster(G, initialOpinions, resistances=resistances1, x_start=x_start, active_nodes=active_nodes, targetNodes=targetNodes)

        current_x_start[stooges[-1] if len(stooges) > 0 else None] = x_start
        return mse

    xs = [(i, r) for i in range(n) for r in [0, 1]] # None will be replaced with x_start
    lazy_greedy(calc_mse, xs, stoogeCount)