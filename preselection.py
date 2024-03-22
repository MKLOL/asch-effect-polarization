import numpy as np
import networkx as nx


def preselect_walk(walk_len, G, k):
    nodes = np.random.choice(G.nodes, k)
    for i, u in enumerate(nodes):
        for _ in range(walk_len): u = np.random.choice(list(G.neighbors(u)))
        nodes[i] = u
    return nodes


preselect_walk_0 = lambda G, k: preselect_walk(0, G, k)
preselect_walk_5 = lambda G, k: preselect_walk(5, G, k)
preselect_walk_10 = lambda G, k: preselect_walk(10, G, k)
preselect_walk_50 = lambda G, k: preselect_walk(50, G, k)


def preselect_centrality(G, k):
    x = nx.eigenvector_centrality(G)
    keys = np.array(list(x.keys()))
    values = np.array(list(x.values()))
    return np.random.choice(keys, p=values / np.sum(values), size=k)
