import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mse_graph_calculator import *



def greedyResistanceNegative(G, initialOpinions, stoogeCount, baseResistance=0.5, change_nodes=None, targetNodes = None, verbose=True, positive=True, nodesToTest=None, return_xs=False, polarization=True, initRes=None, nrange = 2):

    theta = None if polarization else np.mean(initialOpinions)

    n = len(G.nodes)
    resistances = baseResistance * np.ones(n)
    if (initRes is not None):
        resistances = np.array(list(initRes))
    targetNodeSet = set()
    if targetNodes is not None:
        targetNodeSet = set(targetNodes)
    # s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
    stoogeDict = {}
    change_nodes = G.nodes if change_nodes is None else change_nodes

    x_start = None
    active_nodes = G.nodes

    mse0, x_start = approximateMseFaster(G, initialOpinions, resistances=resistances, x_start=x_start, active_nodes=active_nodes, targetNodes=targetNodes, theta=theta)
    xs = [x_start]
    mse0s = [mse0]
    stooges = []
    allVals = []
    for i in range(stoogeCount):
        mse_max = mse0
        x_max = None
        r_max = None
        finalVals = []
        for x in change_nodes:
            vals = []
            if x in stoogeDict: continue
            if x in targetNodeSet: continue
            for r in [x/nrange for x in range(nrange+1)]:

                resistances1 = np.copy(resistances)
                resistances1[x] = r

                mse1, _ = approximateMseFaster(G, initialOpinions, resistances=resistances1, x_start=x_start, active_nodes=[x], targetNodes=targetNodes, theta=theta)
                vals.append(mse1)
                if positive:
                    if mse1 < mse_max:
                        mse_max = mse1
                        x_max = x
                        r_max = r
                else:
                    if mse1 > mse_max:
                        mse_max = mse1
                        x_max = x
                        r_max = r
            if x_max == x:
                finalVals = vals
            #if verbose: print(".", end="", flush=True)
            # if (n - x) % 10 == 0: print(f"    {n - x} remaining")

        if x_max is None:
            #print("x_max is none, thus breaking")
            break

        resistances[x_max] = r_max
        stoogeDict[x_max] = True
        stooges.append((r_max, x_max))
        print(finalVals)
        print(finalVals == sorted(finalVals))
        print(r_max, "!!!!!")
        if r_max != 0 and r_max != 1:
            plt.plot(finalVals)
            plt.show()

        allVals.append(finalVals)
        mse0, x_start = approximateMseFaster(G, initialOpinions, resistances=resistances, x_start=x_start, active_nodes=active_nodes, targetNodes=targetNodes, theta=theta)
        xs.append(x_start)
        mse0s.append(mse0)
        #if verbose: print(f"\nIteration {i}: MSE={mse0} (setting resistance({x_max})={r_max})")

    if return_xs: return xs
    return stooges, resistances, mse0s # resistances, mse_max



def lazy_greedy(f, xs, k, minimize=False, epsilon = 1.1):
    n = len(xs)
    picked = []
    current_val = f(picked)

    if minimize:
        marginal_gains = -np.inf * np.ones(n)
    else:
        marginal_gains = np.inf * np.ones(n)
    vals = [current_val]
    for i in range(k):
        best_j = None
        prev_gain = 0
        if minimize:
            sortedVals = enumerate(np.argsort(marginal_gains))
        else:
            sortedVals = enumerate(np.argsort(marginal_gains)[::-1])
        for r, j in sortedVals:
            if xs[j] in picked: continue
            if minimize and marginal_gains[j] >= prev_gain * epsilon: break
            if not minimize and marginal_gains[j] <= prev_gain / epsilon: break
            gain = f(picked + [xs[j]]) - current_val
            marginal_gains[j] = gain
            if not minimize and (best_j is None or gain > marginal_gains[best_j]): best_j = j
            if minimize and (best_j is None or gain < marginal_gains[best_j]): best_j = j
            prev_gain = gain

        print("")
        if best_j is None:
            print("no further improvement possible")
            break
        picked.append(xs[best_j])
        current_val += marginal_gains[best_j]
        vals.append(current_val)

    return picked, vals



def greedyResistance(G, initialOpinions, stoogeCount, baseResistance=0.5, change_nodes=None, targetNodes = None, initRes = None, verbose=True, minimize = False, polarization=True, epsilon=1.1, eps=1e-5):
    theta = None if polarization else np.mean(initialOpinions)

    n = len(G.nodes)
    resistances = baseResistance * np.ones(n)
    if initRes is not None:
        resistances = np.array(list(initRes))

    current_x_start = {}

    def calc_mse(stooges):
        resistances1 = np.copy(resistances)
        for x, r in stooges:
            resistances1[x] = r

        x_start = current_x_start.get(stooges[-2] if len(stooges) > 1 else None, None)
        active_nodes = [stooges[-1][0]] if len(stooges) > 0 else None

        mse, x_start = approximateMseFaster(G, initialOpinions, resistances=resistances1, x_start=x_start, active_nodes=active_nodes, targetNodes=targetNodes, theta=theta, eps=eps)

        current_x_start[stooges[-1] if len(stooges) > 0 else None] = x_start
        return mse

    xs = [(i, r) for i in range(n) for r in [0, 1]]
    stooges, intermediateMSEs = lazy_greedy(calc_mse, xs, stoogeCount, minimize=minimize, epsilon=epsilon)
    for i, r in stooges: resistances[i] = r
    return [current_x_start[stooge] for stooge in [None] + stooges], resistances

