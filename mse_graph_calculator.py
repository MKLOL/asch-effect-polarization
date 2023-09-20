import numpy as np
import networkx as nx

def calculateMse(G, slist, numCompute = 10000):
    n = len(G.nodes)
    W = nx.adjacency_matrix(G).toarray()
    W = W / np.sum(W, axis=0)[:, None]

    A = np.diag([0.01] * n)

    I = np.eye(n)
    X = np.linalg.pinv(I - (I - A) @ W) @ A

    s_val = []
    x_val = []
    for _ in range(numCompute):
        s = None
        if slist is None:
            s = np.random.normal(0, 1, n)
        else:
            s = slist
        x = X @ s

        s_val.append(np.mean(s) ** 2)
        x_val.append(np.mean(x) ** 2)

    s_mse = np.mean(s_val)
    x_mse = np.mean(x_val)

    return (s_mse, x_mse)