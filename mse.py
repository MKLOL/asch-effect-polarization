from mse_stooges_greedy import *
from mse_graph_calculator import *

n = 20

G = nx.star_graph(n)
G.add_edges_from(zip(G.nodes, G.nodes))

d, ls = getStoogeList(G, 20)

print(calculateMse(G, ls, 1))
