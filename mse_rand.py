from mse_stooges_greedy import *
from mse_graph_calculator import *
from mse_edges_greedy import *
from mse_stooges_resistance_greedy import *
from mse_stooges_mega_greeedy import *
import networkx as nx
import matplotlib.pyplot as plt


n = 100

p = 0.01
G = nx.erdos_renyi_graph(n, p=p)
G.add_edges_from(zip(G.nodes, G.nodes))



walk_len = 100
yss = []

for _ in range(5):
    change_nodes = np.random.choice(G.nodes, 20)
    for i, u in enumerate(change_nodes):
        for _ in range(walk_len): u = np.random.choice(list(G.neighbors(u)))
        change_nodes[i] = u

    ys = greedyResistance(G, len(change_nodes), change_nodes=change_nodes)
    yss.append(ys)

maxlen = max(map(len, yss))
yss = [ys + [ys[-1]] * (maxlen - len(ys)) for ys in yss]

yss = np.array(yss)
ys_mean = np.mean(yss, axis=0)
ys_std = np.std(yss, axis=0) / 100

xs = np.arange(1, len(ys_mean) + 1)
plt.plot(xs, ys_mean)
plt.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha=0.2)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title(f"GNP({n}, {p}) (walk_len={walk_len})")
plt.savefig(f"plots/gnp-{walk_len}.pdf")


