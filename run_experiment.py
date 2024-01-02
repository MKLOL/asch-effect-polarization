"""

Donwload and unzip the datasets into a folder called datasets

"""


import networkx as nx
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from mse_stooges_resistance_greedy import *


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

def apply_greedy(G, num_stooges=50):
    attr = nx.get_node_attributes(G, INTERNAL_OPINION)
    initialOpinions = np.empty(len(attr))
    initialOpinions[list(attr.keys())] = list(attr.values())
    greedyResistance(G , initialOpinions, num_stooges)


if len(sys.argv) > 1:
    G = read(sys.argv[1])
    apply_greedy(G)
else:
    import subprocess
    for file in subprocess.check_output("find . -name '*.tsv' ! -name '*_*'", shell=True).decode("utf-8").split("\n"):
        file = file.strip()
        if not file: continue
        print(">>>", file)
        G = read(file)
        apply_greedy(G)

