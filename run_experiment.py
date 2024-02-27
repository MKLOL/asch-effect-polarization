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
    print("minimze", minimize)
    return greedyResistance(G, initialOpinions, num_stooges, minimize=minimize)


def apply_greedy_opin(G, res, opin, num_stooges=50, minimize=False):
    return greedyResistance(G , opin, num_stooges, initRes=res, minimize=minimize)

if len(sys.argv) > 1:
    G = read(sys.argv[1])
    print(G)
    apply_greedy(G, minimize=True)
else:
    import subprocess
    skip = True
    for file in subprocess.check_output("find . -name '*.tsv' ! -name '*[12]*'", shell=True).decode("utf-8").split("\n"):
        # if "russia_march" not in file: continue
        file = file.strip()
        if not file: continue
        name = re.findall("\w+/\w+(?=\.tsv)", file)[0].replace("/", "-")
        print(">>>", file, name)
        G = read(file)
        nx.write_graphml(G, f"graphml/{name}.graphml")
        num_stooges = int(5 * np.log2(len(G.nodes)))
        print(f"using up to {num_stooges} stooges")

        xs = apply_greedy(G, num_stooges=num_stooges, minimize=True)[0]
        statss = []
        for i, x in enumerate(xs):
            stats = experiment_helpers.record_stats(x, name, f"post-{i}")
            statss.append(stats)

        experiment_helpers.plot_change(statss, name)
