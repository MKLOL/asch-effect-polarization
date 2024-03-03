import networkx as nx
import graph_construction
import random
import numpy as np
from networkx import bipartite

def create_d_regular_bipartite_graph(m, n, d):
    # Check if d-regular bipartite graph is possible
    if m * d != n * d:
        raise ValueError("It's not possible to create a d-regular bipartite graph with these parameters.")

    # Create empty graph
    B = nx.Graph()

    # Add nodes with bipartite sets
    B.add_nodes_from(range(m), bipartite=0)  # Add nodes for partition U
    B.add_nodes_from(range(m, m + n), bipartite=1)  # Add nodes for partition V

    # Manually add edges to satisfy d-regular condition
    for i in range(m):
        for j in range(d):
            B.add_edge(i, (i + j) % n + m)

    if not bipartite.is_bipartite(B):
        raise Exception("Failed to create a bipartite graph.")

    # Check if the graph is d-regular
    if not all([d == B.degree(n) for n in range(m + n)]):
        raise Exception("Failed to create a d-regular bipartite graph.")

    return B

def getGraph(type, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


    s = []
    n = 150
    if type == "GNP":
        G = nx.erdos_renyi_graph(n, 0.05)
    elif type == "d_regular":
        G = create_d_regular_bipartite_graph(n//2, n - n // 2, 20)
    elif type == "star_random" or type == "STAR":
        G = nx.star_graph(n)
    elif type == "tree":
        G = nx.random_tree(n)
    elif type == "smallCommunities":
        G, s = graph_construction.makeGraphFromFile("smallCommunities.in")

    G.add_edges_from(zip(G.nodes, G.nodes))
    n = len(G.nodes)
    if type == "GNP" or type == "star_random" or type == "tree":
        s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)


    if type == "STAR":
        s.append(1)
        for x in range(n - 1):
            s.append(0.498433)
    if type == "d_regular":
        s = np.clip(np.random.normal(0.8, 0.5, n // 2), 0, 1)
        y = np.clip(np.random.normal(0.2, 0.5, n // 2), 0, 1)
        s = np.append(s, y)
    return G, s

def write_statistics_to_file(filename, data):
    # Perform computations
    data_array = np.array(data)
    mse = np.mean((data_array - np.mean(data_array)) ** 2)
    average = np.mean(data_array)
    total_sum = np.sum(data_array)
    median = np.median(data_array)
    quantiles = np.percentile(data_array, [25, 50, 75])

    # Open the file once all computations are done
    with open(filename, 'w') as file:
        file.write(f'Mean Squared Error: {mse}\n')
        file.write(f'Average: {average}\n')
        file.write(f'Sum: {total_sum}\n')
        file.write(f'Median: {median}\n')
        file.write(f'Quantiles (25th, 50th, 75th): {quantiles}\n')