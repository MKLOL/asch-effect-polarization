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
graph_type_labels = {
    "GNP": "GNP$(150, 0.05)$",
    "grid": "Grid",
    "smallCommunities": "RndCommunities",
    "tree": "RndTree$(150)$",
}
init_type_labels = {
    "uniform": "uniform",
    "gaussian": "normal",
    "exponential": "exponential",
}



def apply_greedy(G, s, num_stooges, minimize, method, resistances=None, polarization=True, return_xs=True, phi=1.1, eps=1e-5):
    if method == "greedy":
        ans = greedyResistance(G, s, num_stooges, minimize=minimize, initRes=resistances, polarization=polarization, epsilon=phi, eps=eps)
        xs = ans[0] if return_xs else ans

    elif method =="naive-greedy":
        xs = greedyResistanceNegative(G, s, num_stooges, positive=minimize, initRes=resistances, return_xs=return_xs, polarization=polarization)

    elif method == "maxdeg":
        h = dict(G.degree)
        ls = sorted([(h[x], x) for x in h])[::-1]
        nodes = [x[1] for x in ls]
        nodes = nodes[:num_stooges]
        xs = greedyResistanceNegative(G, s, num_stooges, positive=minimize, change_nodes=nodes, initRes=resistances, return_xs=return_xs, polarization=polarization)

    elif method == "random":
        nodes = list(G.nodes)
        random.shuffle(nodes)
        nodes = nodes[:num_stooges]
        xs = greedyResistanceNegative(G, s, num_stooges, positive=minimize, change_nodes=nodes, initRes=resistances, return_xs=return_xs, polarization=polarization)

    elif method == "centrality":
        h = nx.betweenness_centrality(G)
        ls = sorted([(h[x], x) for x in h])[::-1]
        nodes = [x[1] for x in ls]
        nodes = nodes[:num_stooges]
        xs = greedyResistanceNegative(G, s, num_stooges, positive=minimize, change_nodes=nodes, initRes=resistances, return_xs=return_xs, polarization=polarization)

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

    for method, df in df.groupby("method", sort=False):
        label = method_labels[method]
        x = df.groupby("n")
        mean = x["time"].mean()
        std = x["time"].std()

        plt.plot(mean.index, mean, label=label, **next_config()) # markevery=5
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.yscale("log")
    plt.ylim(None, 5000)
    plt.legend(loc="upper left", ncol=2)
    plt.xlabel("$n$")
    plt.ylabel("time [seconds]")
    savefig()


@memoize
def brute_force(n, num_stooges, seed=None):
    import experiment_std
    G, s = experiment_std.getGraph("GNP", seed, n=n)
    l1, l2 = experiment_std.getRandomBrut(G, s, num_stooges)
    print(">>>2", l1[-1] - l2[-1])
    return {"greedy": l1, "brute": l2}


@genpath
def plot_brute_force(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: brute_force(row.n, row.num_stooges, seed=row.seed))

    num_stooges = setup["num_stooges"][0]
    greedy = np.array([xs + [xs[-1]]*(1 + num_stooges - len(xs)) for xs in df["greedy"]])
    brute = np.array([xs + [xs[-1]]*(1 + num_stooges - len(xs)) for xs in df["brute"]])

    for _ in range(3): next_config()

    for label, xs in [("Greedy", greedy), ("Brute Force", brute)]:
        # import pdb; pdb.set_trace()
        mean = np.nanmean(xs, axis=0)
        std = np.nanstd(xs, axis=0)
        plt.plot(range(num_stooges + 1), mean, label=label, **next_config())
        plt.fill_between(range(num_stooges + 1), mean - std, mean + std, alpha=0.2)

    plt.xlabel("\#stooges")
    plt.ylabel("MSE")
    plt.legend()
    savefig()


@memoize
def isect_min_max(graph_type, polarization, seed=None):
    G, s = graph_creator.getGraph(graph_type, seed=seed)

    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), polarization=polarization, positive=True)
    stoogeNeg, resistNeg, lsNeg = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), polarization=polarization, positive=False)

    sps = set([x[1] for x in stoogePos])
    sns = set([x[1] for x in stoogeNeg])

    jaccard = len(sps.intersection(sns)) / len(sps.union(sns))
    return {"jaccard": jaccard}

def test_isect_min_max(setup):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: isect_min_max(row.graph_type, row.polarization, seed=row.seed))

    grp = df.groupby("graph_type")
    print(grp.mean())
    print(grp.std())


@memoize
def isect_pol_mse(graph_type, minimize, num_stooges=None, seed=None):
    G, s = graph_creator.getGraph(graph_type, seed=seed)

    num_stooges = num_stooges or int(math.log2(len(G.nodes)) * 5)
    _, resistances_mse = apply_greedy(G, s, num_stooges, minimize, "greedy", return_xs=False, polarization=False)
    _, resistances_pol = apply_greedy(G, s, num_stooges, minimize, "greedy", return_xs=False, polarization=True)

    stooges_mse = set(np.where(resistances_mse != 0.5)[0])
    stooges_pol = set(np.where(resistances_pol != 0.5)[0])

    # stooge_pol, resistPos, lsPos = greedyResistanceNegative(G, s, num_stooges, polarization=True, positive=minimize)
    # stooge_mse, resistNeg, lsNeg = greedyResistanceNegative(G, s, num_stooges, polarization=False, positive=minimize)

    # sps = set([x[1] for x in stooge_pol])
    # sns = set([x[1] for x in stooge_mse])

    # jaccard = len(sps.intersection(sns)) / len(sps.union(sns))

    _, xs_pol = approximateMseFaster(G, s, resistances=resistances_pol, eps=1e-10)
    _, xs_mse = approximateMseFaster(G, s, resistances=resistances_mse, eps=1e-10)
    true_mean = np.mean(s)
    mse_pol = np.mean((xs_pol - true_mean)**2)
    mse_mse = np.mean((xs_mse - true_mean)**2)
    pol_pol = np.var(xs_pol)
    pol_mse = np.var(xs_mse)

    jaccard = len(stooges_mse.intersection(stooges_pol)) / len(stooges_mse.union(stooges_pol))
    return {"jaccard": jaccard, "mse_mse": mse_mse, "mse_pol": mse_pol, "pol_mse": pol_mse, "pol_pol": pol_pol}

def test_isect_pol_mse(setup):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: isect_pol_mse(row.graph_type, row.minimize, seed=row.seed))

    grp = df.groupby("graph_type")
    print(grp.mean())
    print(grp.std())
    import pdb; pdb.set_trace()

@genpath
def test_isect_pol_mse_change(setup, has_legend=True, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: isect_pol_mse(row.graph_type, row.minimize, num_stooges=row.num_stooges, seed=row.seed))

    for graph_type, df in df.groupby("graph_type"):
        label = graph_type_labels[graph_type]
        x = df[["num_stooges", "jaccard"]].groupby("num_stooges")["jaccard"]
        mean = x.mean()
        std = x.std()

        plt.plot(mean.index, mean, label=label, **next_config()) # markevery=5
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.ylabel("Jaccard index")
    plt.xlabel("\#stooges")
    if has_legend: plt.legend()
    savefig(f"{'min' if setup['minimize'][0] else 'max'}")


@memoize
def test_algo(graph_type, num_stooges, minimize, eps, phi, n, seed=None):
    G, s = graph_creator.getGraph(graph_type, seed=seed, n=n)
    start_time = time.time()
    _, resistances = apply_greedy(G, s, num_stooges, minimize, "greedy", resistances=None, return_xs=False, polarization=False, eps=eps, phi=phi)
    t = time.time() - start_time
    _, xs = approximateMseFaster(G, s, resistances=resistances, eps=1e-10)
    true_mean = np.mean(s)
    mse = np.mean((xs - true_mean)**2)
    return {"mse": mse, "time": t}

@genpath
def plot_algo(setup, group, ylim=(None, None), savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_algo(row.graph_type, row.num_stooges, row.minimize, row.eps, row.phi, row.n, seed=row.seed),
                 axis=1, result_type='expand'))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    mean0 = None
    for marker, color, (eps, df) in zip(["s", "v", "o", "p"], ["black", "tab:pink", "tab:cyan", "tab:brown"], df.groupby(group, sort=False)):
        eps_label = {1e-5: "10^{-5}", 1e-8: "0", 100.0: "\\infty"}.get(eps, eps)
        label = f"$\\{'epsilon' if group == 'eps' else 'phi'}={eps_label}$"
        grp_mse = df.groupby("num_stooges")["mse"]
        mean = grp_mse.mean()
        if mean0 is None: mean0 = mean
        std = grp_mse.std()
        config = next_config()
        config["color"] = color
        config["marker"] = marker
        ax1.plot(mean.index, mean0 - mean, label=label, **config) # markevery=5
        # ax1.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
        grp_time = df.groupby("num_stooges")["time"]
        mean = grp_time.mean()
        ax2.plot(mean.index, mean, **config, linestyle="dashed") # markevery=5

    # ax1.set_zorder(1)
    # ax1.set_frame_on(False)
    # ax2.set_frame_on(True)

    from matplotlib.lines import Line2D
    line1 = Line2D([0], [0], label='MSE', color='black')
    line2 = Line2D([0], [0], label='time', color='black', linestyle="dashed")
    ax2.legend(handles=[line1, line2], loc="upper right")

    ax1.set_ylim(None, ylim[0])
    ax2.set_ylim(None, ylim[1])
    ax1.set_ylabel("MSE loss")
    ax2.set_ylabel("time [seconds]")
    ax1.set_xlabel("\#stooges")
    ax1.legend(loc="upper left")
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    savefig(f"{setup['graph_type'][0]}-{'min' if setup['minimize'][0] else 'max'}-{group}")



def getDistGraph(G, s, num_stooges, minimize, method, polarization):
    if method == "greedy": method = "naive-greedy"
    stoogePos, resistPos, lsPos = apply_greedy(G, s, num_stooges, minimize, method, resistances=None, return_xs=False, polarization=polarization)
    sps = set([x[1] for x in stoogePos])
    layers = nx.bfs_layers(G, sps)
    ret = dict()
    _, x = approximateMseFaster(G, s, resistances=resistPos)
    layered_x = []
    for layer in layers:
        layered_x.append(x[layer])
    mean = np.mean(x)
    return layered_x, mean


# @memoize
def dists_test(graph_type, num_stooges, minimize, method, polarization, seed=None):
    G, s = graph_creator.getGraph(graph_type, seed=seed)
    if num_stooges is None: num_stooges = int(math.log2(len(G.nodes)) * 5)
    layered_x, mean = getDistGraph(G, s, num_stooges, minimize, method, polarization)
    return {"layered_x": layered_x, "mean": mean}


@genpath
def dists_plot(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: dists_test(row.graph_type, row.num_stooges, row.minimize, row.method, row.polarization, seed=row.seed),
                 axis=1, result_type='expand'))

    num_methods = len(setup["method"])
    for i, (method, df) in enumerate(df.groupby("method", sort=False)):
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
    savefig('min' if setup['minimize'][0] else 'max'-{'pol' if setup['polarization'][0] else 'mse'})


@memoize
def synthetic(graph_type, init_type, num_stooges, minimize, method, polarization, normalize_var=False, seed=None):
    G, s = graph_creator.getGraph(graph_type, seed=seed)
    n = len(G.nodes)
    if init_type is not None:
        std = np.sqrt(1/12)
        if init_type == "uniform":
            s = np.random.random(n) # var = 1/12
        elif init_type == "gaussian":
            s = np.random.normal(0.5, std if normalize_var else 0.5, n)
        elif init_type == "exponential":
            s = np.random.exponential(std if normalize_var else 1.0, n)
        else:
            assert(False)

    xs = apply_greedy(G, s, num_stooges, minimize, method, polarization=polarization)
    return {"s": s, "fst": xs[0], "lst": xs[-1], "xs": xs}


def plot_mse(df, show_var=False, ax=plt, show_decomp=False, label=None):
    num_stooges = max([0] + [len(xs) for xs in df["xs"] if xs not in [None, np.nan]]) # max(map(len, df["xs"]))
    rng = range(num_stooges)
    s = df["s"].iloc[0]
    true_mean = np.mean(s)
    MSE = np.empty((len(df), num_stooges))
    MSE[:] = np.nan
    VAR = np.empty((len(df), num_stooges))
    VAR[:] = np.nan
    BIAS = np.empty((len(df), num_stooges))
    BIAS[:] = np.nan
    for i, xs in enumerate(df["xs"]):
        if xs is None: continue
        for j in range(num_stooges):
            x = xs[j] if j < len(xs) else xs[-1]
            VAR[i, j] = np.var(x)
            MSE[i, j] = np.mean((x - true_mean)**2)
            BIAS[i, j] = (np.mean(x) - true_mean)**2

    mean_VAR = np.nanmean(VAR, axis=0)
    std_VAR = np.nanstd(VAR, axis=0)

    show = ([MSE] if show_decomp or not show_var else []) + \
           ([VAR] if show_decomp or show_var else []) + \
           ([BIAS] if show_decomp else [])

    if len(show) == 1: label = [label]

    colors = ["tab:pink", "tab:cyan", "tab:brown"]
    for color, label, X in zip(colors, label, show):
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        config = next_config()
        if show_decomp: config["color"] = color
        ax.plot(rng, mean, label=label, **config, markevery=5)
        ax.fill_between(rng, mean - std, mean + std, alpha=0.2)


@genpath
def plot_synthetic(setup, has_legend=True, side_by_side=False, title=None, savefig=None):
    global current_markers, current_colors

    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: synthetic(row.graph_type, row.init_type, row.num_stooges, row.minimize, row.method, row.polarization, **({"normalize_var": row.normalize_var} if "normalize_var" in row else {}), seed=row.seed),
                 axis=1, result_type='expand'))

    grp = df.fillna("-").groupby("init_type", sort=False)
    _, axs = plt.subplots(1, len(grp), sharey=True, figsize=((2 if side_by_side else 1) * 6.1, 4.0))
    axs = axs if side_by_side else [axs]
    for ax, (init_type, df) in zip(axs, grp):
        reset_config()
        for method, df in df.groupby("method", sort=False):
            plot_mse(df, show_var=setup["polarization"][0], label=method_labels[method], ax=ax)

        ax.set_xlabel("\#stooges")
        if init_type != "-": ax.set_title(init_type_labels[init_type], fontsize=16)
        # plt.subplots_adjust(wspace=0, hspace=0)

    axs[0].set_ylabel('Polarization' if setup['polarization'][0] else 'MSE')
    if has_legend: axs[0].legend()
    if title is not None: plt.title(f"{'Minimizing' if setup['minimize'][0] else 'Maximizing'} {'polarization' if setup['polarization'][0] else 'MSE'} for {title}", fontsize=16)

    init_type = "" if setup["init_type"][0] is None else f"-{setup['init_type'][0]}"
    savefig(f"{'ALL' if side_by_side else setup['graph_type'][0]}-{'min' if setup['minimize'][0] else 'max'}{init_type}-{'pol' if setup['polarization'][0] else 'mse'}")


@genpath
def plot_synthetic_opinions(setup, title=None, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: synthetic(row.graph_type, row.init_type, row.num_stooges, row.minimize, row.method, row.polarization, seed=row.seed),
                 axis=1, result_type='expand'))

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(6.1, 5.0))
    pol_row = df[df["polarization"]].iloc[0]
    mse_row = df[~df["polarization"]].iloc[0]

    for x, x_label, ax, keep_axis in [(mse_row["s"], "$s$", ax1, False), (mse_row["fst"], "$x^*$", ax2, False), (mse_row["lst"], "$x^*(\\textrm{MSE})$", ax3, False), (pol_row["lst"], "$x^*(\\textrm{polarization})$", ax4, True)]:
        text = ax.set_title(x_label, y=1.0, pad=-20)
        import matplotlib.patheffects as path_effects
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
        ax.set_xlabel("opinions")
        if not keep_axis: ax.get_xaxis().set_visible(False)
        ax.hist(x, bins=20, edgecolor='white', range=[0, 1])

    if title is not None: plt.suptitle(f"{'Minimization' if setup['minimize'][0] else 'Maximization'} for {title}", fontsize=16, y=0.92)
    savefig(f"{setup['graph_type'][0]}-{'min' if setup['minimize'][0] else 'max'}-{'pol' if setup['polarization'][0] else 'mse'}")


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
def real_world(name, minimize, method, polarization, seed=None):
    import tweet_loader

    if name == "vax":
        G, resistances, initialOpinions = tweet_loader.getTweetData("vax_gpt_labels.pkl")

    elif name == "war":
        G, resistances, initialOpinions = tweet_loader.getTweetData("war_gpt_labels.pkl")

    else:
        a, b = name.split("-")
        file = f"./datasets/{a}/{b}/{a}.tsv"
        print(">>>", file, name)
        G = read(file)
        print(f"num nodes={len(G.nodes)}")

        attr = nx.get_node_attributes(G, INTERNAL_OPINION)
        initialOpinions = np.empty(len(attr))
        initialOpinions[list(attr.keys())] = list(attr.values())
        resistances = None

    # nx.write_graphml(G, f"graphml/{name}.graphml")
    num_stooges = int(5 * np.log2(len(G.nodes)))
    print(f"using up to {num_stooges} stooges")

    if len(G.nodes) > 10000 and method == "centrality":
        print(f"ABORTING CENTRALITY BECAUSE {len(G.nodes)} ARE TOO MUCH")
        return None

    xs = apply_greedy(G, initialOpinions, num_stooges, minimize=minimize, method=method, resistances=resistances, polarization=polarization)
    return {"s": initialOpinions, "fst": xs[0], "lst": xs[-1], "xs": xs}


@genpath
def plot_real_world_opinions(setup, show_diff=False, title=True, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: real_world(row.dataset, row.minimize, row.method, row.polarization, seed=row.seed),
                 axis=1, result_type='expand'))

    """
    _, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6.1, 4.5))
    for x_name, x_label, ax, keep_axis in [("s", "$s$", ax1, False), ("fst", "$x^*$", ax2, False), ("lst", "$x^*_{\\textrm{stooge}}$", ax3, True)]:
        x = df[x_name].iloc[0]

        ax.set_title(x_label, y=1.0, pad=-20)
        ax.set_xlabel("opinions")
        if not keep_axis: ax.get_xaxis().set_visible(False)
        ax.hist(x, bins=20, edgecolor='white', range=[0, 1])

    savefig(f"{setup['dataset'][0]}-{'min' if setup['minimize'][0] else 'max'}-{'pol' if setup['polarization'][0] else 'mse'}")
    """

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(6.1, 5.0))
    pol_row = df[df["polarization"]].iloc[0]
    mse_row = df[~df["polarization"]].iloc[0]

    x0 = mse_row["fst"]

    for x, x_label, ax, keep_axis, use_diff in [(mse_row["s"], "$s$", ax1, False, False), (mse_row["fst"], "$x^*$", ax2, False, False), (mse_row["lst"], "$x^*(\\textrm{MSE})$", ax3, False, True), (pol_row["lst"], "$x^*(\\textrm{polarization})$", ax4, True, True)]:
        text = ax.set_title(x_label, y=1.0, pad=-20)
        import matplotlib.patheffects as path_effects
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
        ax.set_xlabel("opinions")
        if not keep_axis: ax.get_xaxis().set_visible(False)
        if show_diff and use_diff:
            hist_vals, _ = np.histogram(x, 20, range=[0, 1])
            hist_vals0, _ = np.histogram(x0, 20, range=[0, 1])
            hist_diff = hist_vals - hist_vals0
            rng = (0.5 + np.arange(20)) / 20
            for ix, color in [(hist_diff > 0, "green"), (hist_diff < 0, "red")]:
                ax.hist(rng[ix], bins=20, range=[0, 1], weights=hist_diff[ix], color=color)
            # import pdb; pdb.set_trace()
        else:
            ax.hist(x, bins=20, edgecolor='white', range=[0, 1])
        # m = max(np.histogram(x, 20, range=(0,1))[0])
        # ax.set_yticks([0, m])

    if title: plt.suptitle(f"{'Minimization' if setup['minimize'][0] else 'Maximization'} for {setup['dataset'][0].title()}", fontsize=16, y=0.92)
    savefig(f"{setup['dataset'][0]}-{'min' if setup['minimize'][0] else 'max'}")



@genpath
def plot_real_world_change(setup, has_legend=True, show_decomp=False, title=True, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = papply(df, lambda row: real_world(row.dataset, row.minimize, row.method, row.polarization, seed=row.seed))
    # df = df.join(df.astype("object").apply(lambda row: real_world(row.dataset, row.minimize, row.method, seed=row.seed),
    #              axis=1, result_type='expand'))

    for method, df in df.groupby("method", sort=False):
        plot_mse(df, show_var=setup["polarization"][0], label=["MSE", "Polarization", "Bias$^2$"] if show_decomp else method_labels[method], show_decomp=show_decomp)

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

    plt.xlabel("\#stooges")
    if has_legend: plt.legend()
    if not show_decomp: plt.ylabel('Polarization' if setup['polarization'][0] else 'MSE')

    if title: plt.title(f"{'Minimizing' if setup['minimize'][0] else 'Maximizing'} {'polarization' if setup['polarization'][0] else 'MSE'} for {setup['dataset'][0].title()}", fontsize=16)
    savefig(f"{setup['dataset'][0]}-{'min' if setup['minimize'][0] else 'max'}-{'pol' if setup['polarization'][0] else 'mse'}{'-decomp' if show_decomp else ''}")





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

