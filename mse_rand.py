from mse_stooges_greedy import *
from mse_graph_calculator import *
from mse_edges_greedy import *
from mse_stooges_resistance_greedy import *
from mse_stooges_mega_greeedy import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from preselection import *
import functools


@functools.lru_cache(maxsize=None)
def generate_instance(n, p, seed=None):
    G = nx.erdos_renyi_graph(n, p=p)
    G.add_edges_from(zip(G.nodes, G.nodes))
    s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
    return G, s


def test(n, p, k, preselect, max_repeat=50, eps=0.005):
    yss = []

    for i in range(max_repeat):
        G, s = generate_instance(n, p, seed=i)
        change_nodes = preselect(G, k)
        ys = greedyResistance(G, s, len(change_nodes), change_nodes=change_nodes, verbose=False)
        yss.append(ys)
        if i > 5 and (np.var([ys[-1] for ys in yss]) < 0.1 * i * eps**2): break
        print(f"{i:3d}: {np.std([ys[-1] for ys in yss]) / np.sqrt(0.1 * i) :.10f}")

    maxlen = max(map(len, yss))
    yss = [ys + [ys[-1]] * (maxlen - len(ys)) for ys in yss]

    return np.array(yss)


def add_plot(yss, label, std_scale=0.2):
    yss = test(n, p, k, preselect_walk_0)
    yss_mean = np.mean(yss, axis=0)
    yss_std = np.std(yss, axis=0)
    xs = np.arange(len(yss_mean))

    plt.plot(xs, yss_mean, label=label)
    plt.fill_between(xs, yss_mean - std_scale * yss_std, yss_mean + std_scale * yss_std, alpha=0.2)


n = 100
p = 0.1
k = 100

add_plot(test(n, p, k, preselect_walk_0), "walk_len=0")
add_plot(test(n, p, k, preselect_walk_5), "walk_len=5")
add_plot(test(n, p, k, preselect_walk_10), "walk_len=10")
add_plot(test(n, p, k, preselect_walk_50), "walk_len=50")
add_plot(test(n, p, k, preselect_centrality), "centrality")


plt.xlabel("#Stooges")
plt.ylabel("MSE")
plt.title(f"GNP({n}, {p})")
plt.legend()
plt.savefig(f"plots/stooges-{k}-gnp-{n}-{p}.pdf")


