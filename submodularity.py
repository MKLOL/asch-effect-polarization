import numpy as np
import networkx as nx
import itertools
import pandas as pd


def probs(G):
    n = len(G.nodes)
    W = nx.adjacency_matrix(G).toarray()
    W = W / np.sum(W, axis=0)[:, None]

    A = np.diag(0.5 * np.ones(n))
    P = np.zeros((n, n))

    for i in range(n):
        X = (np.eye(n) - A) @ W
        X -= np.eye(n)
        X[i, :] = 0
        X[i, i] = 1
        b = np.eye(n)[i]

        p = np.linalg.solve(X, b)
        p = np.maximum(p, 0)
        P[i] = p

    return P



n = 10
p = 0.1

G = nx.erdos_renyi_graph(n, p)
# G = nx.barabasi_albert_graph(n, 2)
# G = nx.lollipop_graph(n, 2*n)
# G = nx.grid_2d_graph(n, n)

G.add_edges_from(zip(G.nodes, G.nodes))
P = probs(G)

rows = []
for a, b in itertools.combinations(range(n), 2):
    lhs = 0
    rhs = 0
    lhs_full = 0
    rhs_full = 0

    for v in range(n):
        Pab = P[v, a] * P[a, b]
        Pba = P[v, b] * P[b, a]
        Pa = P[v, a] * (1 - P[a, b])
        Pb = P[v, b] * (1 - P[b, a])
        lhs += (Pab + Pba)**2
        rhs += 2 * Pa * Pb
        lhs_full += (Pa + Pab + Pba)**2 + (Pb + Pab + Pba)**2
        rhs_full += (Pa + Pb + Pab + Pba)**2

    rows.append({
        "a": a,
        "b": b,
        "lhs":lhs,
        "rhs": rhs,
        "lhs_full": lhs_full,
        "rhs_full": rhs_full
        })

df = pd.DataFrame(rows)

df["ratio_full"] = df.lhs_full / df.rhs_full
print(df[["lhs_full", "rhs_full", "ratio_full"]].describe())

"""
ix = (df.lhs_full / df.rhs_full).argmin()
lhs, rhs, lhs_full, rhs_full = df.loc[ix]

print("reduced:", lhs, ">=", rhs)
print(f"ratio={lhs / rhs}")
print(f"density={2 * m / n / (n - 1)}")

print("\nfull:", lhs_full, ">=", rhs_full)
print(f"ratio={lhs_full / rhs_full}")
"""
