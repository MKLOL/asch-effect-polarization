from mse_stooges_greedy import *
from mse_graph_calculator import *
from mse_edges_greedy import *
from mse_stooges_resistance_greedy import *
from mse_stooges_mega_greeedy import *
import networkx as nx
"""
n = 44

G = nx.erdos_renyi_graph(n, p=0.3)
G.add_edges_from(zip(G.nodes, G.nodes))
edgeCount = 10

print(getEdgeList(G, edgeCount))"""

n = 44

# G = nx.erdos_renyi_graph(n, p=0.1)
G = nx.erdos_renyi_graph(n, p=0.3)
G.add_edges_from(zip(G.nodes, G.nodes))

greedyResistanceMega(G, 10)