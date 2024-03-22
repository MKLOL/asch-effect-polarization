"""

To use, download and unzip the datasets into a folder called datasets

"""


import networkx as nx
import pandas as pd
import numpy as np
import sys
from pathlib import Path

import tweet_loader
from mse_stooges_resistance_greedy import *
import experiment_helpers
import re


INTERNAL_OPINION = "internal_opinion"


def read(graph_file):
    p = Path(graph_file)
    group1_file = f"{p.parent}/query_{p.stem}_1.tsv"
    group2_file = f"{p.parent}/query_{p.stem}_2.tsv"

    G = nx.read_weighted_edgelist(graph_file)
    group1 = pd.read_csv(group1_file, sep='\t', header=None)
    group2 = pd.read_csv(group2_file, sep='\t', header=None)

    nx.set_node_attributes(G, 0.5, INTERNAL_OPINION)
    for _, (id, *_) in group1.iterrows():
        G.nodes[id][INTERNAL_OPINION] = 0
    for _, (id, *_) in group2.iterrows():
        G.nodes[id][INTERNAL_OPINION] = 1

    print("graph created...")
    return nx.convert_node_labels_to_integers(G)

def apply_greedy(G, num_stooges=50, minimize=False):
    attr = nx.get_node_attributes(G, INTERNAL_OPINION)
    initialOpinions = np.empty(len(attr))
    initialOpinions[list(attr.keys())] = list(attr.values())
    return greedyResistance(G , initialOpinions, num_stooges, minimize)

def apply_greedy_opin(G, res, opin, num_stooges=50, minimize=False):
    return greedyResistance(G , opin, num_stooges, initRes=res, minimize=minimize)


G, res, opin = tweet_loader.getTweetData("war_gpt_labels.pkl")
print(len(G.nodes))
print(len(G.edges))

G.add_edges_from(zip(G.nodes, G.nodes))
skip = True
name = "test_dragos_max"
print(">>>", name, name)
x = apply_greedy_opin(G, res, opin, num_stooges=0, minimize=False)
print(list(x))
experiment_helpers.record_stats(x, name, "pre")
x = apply_greedy_opin(G, res, opin, num_stooges=10, minimize=False)
print(list(x))
experiment_helpers.record_stats(x, name, "post-10")
x = apply_greedy_opin(G, res, opin, num_stooges=50, minimize=False)
experiment_helpers.record_stats(x, name, "post-50")