import numpy as np
import networkx as nx
import cython
cimport numpy as cnp


def approximate_opinions(G, slist, resistances=None, max_iterations=20, eps=1e-5, x_start=None, active_nodes=None, verbose=False):
    if resistances is None:
        resistances = 0.5 * np.ones(len(slist))
    x = np.copy(slist if x_start is None else x_start)

    if active_nodes is None:
        active_nodes = G.nodes
    active_nodes = set(active_nodes)

    for i in range(max_iterations):
        if verbose: print(f"Iteration {i}: {np.var(x):.7f}  #active={len(active_nodes)}")
        change_nodes = []
        change_x = []

        for u in active_nodes:
            x_u: cython.float = resistances[u] * slist[u]
            x_vs: cython.float = 0
            for v in G.neighbors(u): x_vs += x[v]
            x_u += (1 - resistances[u]) * x_vs / len(G[u])

            change_nodes.append(u)
            change_x.append(x_u)

        for u, x_u in zip(change_nodes, change_x):
            change = abs(x_u - x[u])
            if change > eps:
                for v in G.neighbors(u):
                    active_nodes.add(v)
            else:
                active_nodes.remove(u)
            x[u] = x_u

        if len(active_nodes) == 0: break

    return x

"""
def approximate_opinions(G, slist, resistances=None, max_iterations=20, eps:cython.float=1e-5, x_start=None, active_nodes=None, targetNodes=None, verbose=False):
    if resistances is None:
        resistances = 0.5 * np.ones(len(slist))
    if active_nodes is None:
        active_nodes = G.nodes
    active_nodes = set(active_nodes)
    cdef cnp.ndarray x = slist if x_start is None else x_start

    for i in range(max_iterations):
        if verbose: print(f"Iteration {i}: {np.var(x):.7f}  #active={len(active_nodes)}")
        change_nodes = []
        change_x = []

        for u in set(active_nodes):
            x_u: cython.float = resistances[u] * slist[u]
            x_vs: cython.float = 0
            for v in G.neighbors(u): x_vs += x[v]
            x_u += (1 - resistances[u]) * x_vs / len(G[u])

            change_nodes.append(u)
            change_x.append(x_u)

            change: cython.float = abs(x_u - x[u])
            if change > eps:
                for v in G.neighbors(u):
                    active_nodes.add(v)
            else:
                active_nodes.remove(u)

        for u, x_u in zip(change_nodes, change_x):
            x[u] = x_u

        if len(active_nodes) == 0: break

    # print(x)
    return x
"""




