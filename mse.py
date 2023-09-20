from mse_stooges_greedy import *
from mse_graph_calculator import *

n = 20

G = nx.erdos_renyi_graph(n, p=0.8)

d, ls = getStoogeList(G, 20)

print(calculateMse(G, ls, 1))
