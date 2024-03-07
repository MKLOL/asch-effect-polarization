import networkx as nx
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import itertools
import random
import re
import math
import time
import cvxpy as cp

from utils import *
import tweet_loader
from mse_stooges_resistance_greedy import *
import experiment_helpers
import graph_creator



INTERNAL_OPINION = "internal_opinion"
method_labels = {
    "random": "Random",
    "maxdeg": "MaxDeg",
    "centrality": "Centrality",
    "greedy": "Greedy",
    "naive-greedy": "NaiveGreedy",
}



def apply_greedy(G, s, num_stooges, minimize, method, resistances=None):
    if method == "greedy":
        xs = greedyResistance(G, s, num_stooges, minimize=minimize, initRes=resistances)[0]

    elif method =="naive-greedy":
        xs = greedyResistanceNegative(G, s, num_stooges, positive=minimize, initRes=resistances, return_xs=True)

    elif method == "maxdeg":
        h = dict(G.degree)
        ls = sorted([(h[x], x) for x in h])[::-1]
        nodes = [x[1] for x in ls]
        nodes = nodes[:num_stooges]
        xs = greedyResistanceNegative(G, s, num_stooges, positive=minimize, change_nodes=nodes, initRes=resistances, return_xs=True)

    elif method == "random":
        nodes = list(G.nodes)
        random.shuffle(nodes)
        nodes = nodes[:num_stooges]
        xs = greedyResistanceNegative(G, s, num_stooges, positive=minimize, change_nodes=nodes, initRes=resistances, return_xs=True)

    elif method == "centrality":
        h = nx.betweenness_centrality(G)
        ls = sorted([(h[x], x) for x in h])[::-1]
        nodes = [x[1] for x in ls]
        nodes = nodes[:num_stooges]
        xs = greedyResistanceNegative(G, s, num_stooges, positive=minimize, change_nodes=nodes, initRes=resistances, return_xs=True)

    else:
        assert(False)

    return xs


def opt(n, num_stooges, seed=None):
    # create instance
    G = nx.erdos_renyi_graph(n, 0.5)
    s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
    W = nx.adjacency_matrix(G).toarray()
    W = W / np.sum(W, axis=0)[:, None]

    # variables
    x = cp.Variable(n)
    is_stooge = cp.Variable(n, boolean=True)
    a_mod = cp.Variable(n, nonneg=True)

    # constraints
    A = cp.diag(a_mod)
    constr_eq = x == A @ s + (np.eye(n) - A) @ W @ x
    constr_a1 = a_mod <= 1
    constr_a2 = a_mod == 0.5

    prob = cp.Problem(cp.Minimize(cp.sum(x)),
            [constr_eq, constr_a1, constr_a2])
    prob.solve(solver=cp.GUROBI)

    import pdb; pdb.set_trace()

    print(">>>", x.value)



@memoize
def scalability(n, num_stooges, minimize, method, seed=None):
    G = nx.erdos_renyi_graph(n, 0.05)
    s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
    start_time = time.time()
    apply_greedy(G, s, num_stooges, minimize, method)
    return {"time": time.time() - start_time}


@genpath
def plot_scalability(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: scalability(row.n, row.num_stooges, row.minimize, row.method, seed=row.seed),
                 axis=1, result_type='expand'))

    for method, df in df.groupby("method"):
        label = method_labels[method]
        x = df.groupby("n")
        mean = x["time"].mean()
        std = x["time"].std()

        plt.plot(mean.index, mean, label=label)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.legend()
    plt.xlabel("$n$")
    plt.ylabel("time [seconds]")
    savefig()


def getDistGraph(G, s, stoogeCount):
    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, stoogeCount, positive=True)
    sps = set([x[1] for x in stoogePos])
    layers = nx.bfs_layers(G, sps)
    ret = dict()
    _, x = approximateMseFaster(G, s, resistances=resistPos)
    layered_x = []
    for layer in layers:
        layered_x.append(x[layer])
    mean = np.mean(x)
    return layered_x, mean


@memoize
def dists_test(graph_type, num_stooges, minimize, method, seed=None):
    G, s = graph_creator.getGraph(graph_type, seed=seed)
    if num_stooges is None: num_stooges = int(math.log2(len(G.nodes)) * 5)
    layered_x, mean = getDistGraph(G, s, num_stooges)                           # TODO: add method
    return {"layered_x": layered_x, "mean": mean}


@genpath
def dists_plot(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: dists_test(row.graph_type, row.num_stooges, row.minimize, row.method, seed=row.seed),
                 axis=1, result_type='expand'))

    num_methods = len(setup["method"])
    for i, (method, df) in enumerate(df.groupby("method")):
        label = method_labels[method]
        layered_x = df["layered_x"].iloc[0]
        x_mean = df["mean"].iloc[0]

        mean = [np.mean((x - x_mean)**2) for x in layered_x]
        std = [np.mean((x - x_mean)**2) for x in layered_x]

        ds = np.arange(len(mean)) + (i+1)/(1+num_methods) - 0.5
        plt.bar(ds, mean, label=label, width=1/(1+num_methods), yerr=np.array(std)/10)

    plt.legend()
    plt.xlabel("minimum distance to a stooge")
    plt.ylabel("MSE")
    savefig()


@memoize
def synthetic(graph_type, init_type, num_stooges, minimize, method, seed=None):
    G, s = graph_creator.getGraph(graph_type, seed=seed)
    n = len(G.nodes)
    if init_type is not None:
        if init_type == "uniform":
            s = np.random.random(n)
        elif init_type == "gaussian":
            s = np.random.normal(0.5, n)
        elif init_type == "abc":
            s = np.random.exponential(1, n)
        else:
            assert(False)

    xs = apply_greedy(G, s, num_stooges, minimize, method)
    return {"s": s, "fst": xs[0], "lst": xs[-1], "xs": xs}


def plot_mse(df, label=None):
    num_stooges = max(map(len, df["xs"]))
    rng = range(num_stooges)
    s = df["s"].iloc[0]
    true_mean = np.mean(s)
    MSE = np.empty((len(df), num_stooges))
    MSE[:] = np.nan
    for i, xs in enumerate(df["xs"]):
        for j, x in enumerate(xs):
            MSE[i, j] = np.mean((x - true_mean)**2)

    mean = np.nanmean(MSE, axis=0)
    std = np.nanstd(MSE, axis=0)
    plt.plot(rng, mean, label=label)
    plt.fill_between(rng, mean - std, mean + std, alpha=0.2)


@genpath
def plot_synthetic(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: synthetic(row.graph_type, row.init_type, row.num_stooges, row.minimize, row.method, seed=row.seed),
                 axis=1, result_type='expand'))

    for method, df in df.groupby("method"):
        plot_mse(df, label=method_labels[method])

    plt.legend()
    plt.xlabel("number of stooges")

    savefig(f"{setup['graph_type'][0]}-{'min' if setup['minimize'][0] else 'max'}")


@genpath
def plot_synthetic_opinions(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: synthetic(row.graph_type, row.init_type, row.num_stooges, row.minimize, row.method, seed=row.seed),
                 axis=1, result_type='expand'))

    _, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6.1, 4.5))
    for x_name, x_label, ax, keep_axis in [("s", "$s$", ax1, False), ("fst", "$x^*$", ax2, False), ("lst", "$x^*_{\\textrm{stooge}}$", ax3, True)]:
        x = df[x_name].iloc[0]

        ax.set_title(x_label, y=1.0, pad=-20)
        ax.set_xlabel("opinions")
        if not keep_axis: ax.get_xaxis().set_visible(False)
        ax.hist(x, bins=20, edgecolor='white', range=[0, 1])

    savefig(f"{setup['graph_type'][0]}-{'min' if setup['minimize'][0] else 'max'}")


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


@memoize
def real_world(name, minimize, method, seed=None):
    if name == "vax":
        import tweet_loader
        G, resistances, initialOpinions = tweet_loader.getTweetData()

    else:
        a, b = name.split("-")
        file = f"./datasets/{a}/{b}/{a}.tsv"
        print(">>>", file, name)
        G = read(file)
        print(f"num nodes={len(G.nodes)}")
        # if len(G.nodes) > 9000: return None

        attr = nx.get_node_attributes(G, INTERNAL_OPINION)
        initialOpinions = np.empty(len(attr))
        initialOpinions[list(attr.keys())] = list(attr.values())
        resistances = None

    # nx.write_graphml(G, f"graphml/{name}.graphml")
    num_stooges = int(5 * np.log2(len(G.nodes)))
    print(f"using up to {num_stooges} stooges")

    xs = apply_greedy(G, initialOpinions, num_stooges, minimize=minimize, method=method, resistances=resistances)
    return {"s": initialOpinions, "fst": xs[0], "lst": xs[-1], "xs": xs}


@genpath
def plot_real_world_opinions(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: real_world(row.dataset, row.minimize, row.method, seed=row.seed),
                 axis=1, result_type='expand'))

    _, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6.1, 4.5))
    for x_name, x_label, ax, keep_axis in [("s", "$s$", ax1, False), ("fst", "$x^*$", ax2, False), ("lst", "$x^*_{\\textrm{stooge}}$", ax3, True)]:
        x = df[x_name].iloc[0]

        ax.set_title(x_label, y=1.0, pad=-20)
        ax.set_xlabel("opinions")
        if not keep_axis: ax.get_xaxis().set_visible(False)
        ax.hist(x, bins=20, edgecolor='white', range=[0, 1])

    savefig(f"{setup['dataset'][0]}-{'min' if setup['minimize'][0] else 'max'}")


@genpath
def plot_real_world_change(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: real_world(row.dataset, row.minimize, row.method, seed=row.seed))
    # df = df.join(df.astype("object").apply(lambda row: real_world(row.dataset, row.minimize, row.method, seed=row.seed),
    #              axis=1, result_type='expand'))

    for method, df in df.groupby("method"):
        plot_mse(df, label=method_labels[method])

    """
    xs = df["xs"].iloc[0]
    rng = range(len(xs))
    s = df["s"].iloc[0]
    true_mean = np.mean(s)
    mse = []
    bias = []
    var = []
    for x in xs:
        mse.append(np.mean((x - true_mean)**2))
        bias.append((np.mean(x) - true_mean)**2)
        var.append(np.mean((x - np.mean(x))**2))

    plt.plot(rng, mse, label="MSE")
    plt.plot(rng, bias, label="Bias")
    plt.plot(rng, var, label="Variance")
    """

    plt.legend()
    plt.xlabel("numer of stooges")

    savefig(f"{setup['dataset'][0]}-{'min' if setup['minimize'][0] else 'max'}")





"""
if len(sys.argv) > 1:
    G = read(sys.argv[1])
    print(G)
    apply_greedy(G, minimize=True)
else:
    import subprocess
    skip = True
    for file in subprocess.check_output("find . -name '*.tsv' ! -name '*[12]*'", shell=True).decode("utf-8").split("\n"):
        # if "russia_march" not in file: continue
        real_world(file)
"""

