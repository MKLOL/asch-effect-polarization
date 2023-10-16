from mse_stooges_greedy import *
from mse_graph_calculator import *
from mse_edges_greedy import *
from mse_stooges_resistance_greedy import *
from mse_stooges_mega_greeedy import *
import networkx as nx


G = nx.read_edgelist("twitter_combined.txt")
G = nx.convert_node_labels_to_integers(G)
n = len(G.nodes)

resistances = 0.5 * np.ones(n)
slist = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)

greedyResistance(G, 10)
