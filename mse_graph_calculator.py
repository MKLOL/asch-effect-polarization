import numpy as np
import networkx as nx


# random nodes
# high-degree nodes
# high-influence (? maybe 2nd degree) 


def approximateMseFaster(G, slist, resistances=None, max_iterations=20, eps=1e-5, x_start=None, active_nodes=None, verbose=False):
    n = len(G.nodes)
    x = slist if x_start is None else x_start

    if active_nodes is None:
        active_nodes = G.nodes
    active_nodes = set(active_nodes)

    for i in range(max_iterations):
        if verbose: print(f"Iteration {i}: {np.var(x):.7f}  #active={len(active_nodes)}")
        x_new = np.copy(x)

        for u in set(active_nodes):
            x_u = resistances[u] * slist[u]
            x_vs = 0
            for v in G.neighbors(u): x_vs += x[v]
            x_u += (1 - resistances[u]) * x_vs / len(G[u])
            x_new[u] = x_u

            change = abs(x_new[u] - x[u])
            if change > eps:
                for v in G.neighbors(u):
                    active_nodes.add(v)
            else:
                active_nodes.remove(u)

        x = x_new
        if len(active_nodes) == 0: break

    # print(x)
    x_mse = np.var(x)
    return x_mse, x


def approximateMseFast(G, slist, resistances=None, max_iterations=100, eps=1e-5):
    n = len(G.nodes)
    x = slist

    for i in range(max_iterations):
        x_new = np.empty(n)
        for u in G.nodes:
            x_u = resistances[u] * slist[u]
            x_vs = 0
            for v in G.neighbors(u): x_vs += x[v]
            x_u += (1 - resistances[u]) * x_vs / len(G[u])
            x_new[u] = x_u

        norm = np.linalg.norm(x - x_new)
        norm_inf = np.max(np.abs(x - x_new))
        if norm_inf < eps: break
        x = x_new

        print(f"Iteration {i}: {np.var(x):.7f} (change={norm:.5f})")

    # print(x)
    x_mse = np.var(x)
    return x_mse, x


def approximateMse(G, slist, resistances=None, max_iterations=100, eps=1e-5):
    n = len(G.nodes)
    W = nx.adjacency_matrix(G).toarray()
    W = W / np.sum(W, axis=0)[:, None]

    A = np.diag([0.5] * n if resistances is None else resistances)

    x = slist

    for i in range(max_iterations):
        x_new = A @ slist + (np.eye(n) - A) @ W @ x
        norm = np.linalg.norm(x - x_new)
        norm_inf = np.max(np.abs(x - x_new))
        if norm_inf < eps: break
        x = x_new

        # print(f"Iteration {i}: {norm}")

    # print(x)
    x_mse = np.var(x) # np.mean((x - np.mean(x))**2)
    return x_mse



def calculateMse(G, slist, numCompute = 100, resistances = None):
    n = len(G.nodes)
    W = nx.adjacency_matrix(G).toarray()
    W = W / np.sum(W, axis=0)[:, None]

    A = np.diag([0.5] * n if resistances is None else resistances)

    I = np.eye(n)
    X = np.linalg.pinv(I - (I - A) @ W) @ A

    s_val = []
    x_val = []
    for _ in range(numCompute):
        s = None
        if slist is None:
            s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
        else:
            s = slist
        xs = X @ s
        # print(xs)

        avgs = np.mean(s)
        avgx = np.mean(xs)
        s_val.append(np.mean([(x-avgs) ** 2 for x in s]))
        x_val.append(np.mean([(x-avgx) ** 2 for x in xs]))

    s_mse = np.mean(s_val)
    x_mse = np.mean(x_val)

    return (s_mse, x_mse)

