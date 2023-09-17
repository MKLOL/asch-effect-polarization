import numpy as np
import networkx as nx


n = 20

G = nx.erdos_renyi_graph(n, p=0.8)
# G = nx.star_graph(n-1)

G.add_edges_from(zip(G.nodes, G.nodes))
# M = nx.normalized_laplacian_matrix(G).toarray()
# M_inv = np.linalg.pinv(M)
W = nx.adjacency_matrix(G).toarray()
W = W / np.sum(W, axis=0)[:, None]

A = np.diag([0.01] * n)
# X = M_inv + np.eye(n)
I = np.eye(n)
X = np.linalg.pinv(I - (I - A) @ W) @ A


s_val = []
x_val = []
for _ in range(10000):
    s = np.random.normal(0, 1, n)
    x = X @ s

    s_val.append(np.mean(s)**2)
    x_val.append(np.mean(x)**2)

s_mse = np.mean(s_val)
x_mse = np.mean(x_val)

print("s_mse:", s_mse)
print("x_mse:", x_mse)

