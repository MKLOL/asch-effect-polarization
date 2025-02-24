from datetime import datetime
import pickle
import os
import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import numpy as np
from matplotlib.ticker import MaxNLocator


ncores = int(os.getenv('NSLOTS', default=1))

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

def getGraph(type, seed=None, n=150):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


    s = []
    if type == "GNP":
        G = nx.erdos_renyi_graph(n, 0.15)
    elif type == "d_regular":
        G = create_d_regular_bipartite_graph(n//2, n - n // 2, 20)
    elif type == "star_random" or type == "STAR":
        G = nx.star_graph(n)
    elif type == "tree":
        G = nx.random_tree(n)
    elif type == "smallCommunities":
        G, s = graph_construction.makeGraphFromFile("graph_descriptions/smallCommunities.in")
    elif type == "grid":
        G = nx.grid_graph(dim=(10, n//10))
        G = nx.convert_node_labels_to_integers(G)
    else:
        assert(False)

    G.add_edges_from(zip(G.nodes, G.nodes))
    n = len(G.nodes)
    if type == "GNP" or type == "star_random" or type == "tree" or type == "grid":
        s = [random.choice([0,0.45,0.55,1]) for x in range(n)]


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



def memoize(func, verbose=True):
    path_tmpl = "cache/" + func.__name__ + "{}.pkl"
    def wrapper(*args, **kwargs):
        arg_str = [str(v) for v in args] + [f"{k}-{v}" for k, v in kwargs.items()]
        path = path_tmpl.format("-" + "--".join(arg_str) if len(arg_str) > 0 else "")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            # with open(path, 'a'):
            #     os.utime(path)
            if verbose:
                arg_prt = [str(v) for v in args] + [f"{k}={v}" for k, v in kwargs.items()]
                print(f"{func.__name__}({', '.join(arg_prt)}) >> {path}")
            cache = func(*args, **kwargs)
            with open(path, "wb") as f:
                pickle.dump(cache, f)
            return cache
    return wrapper

papply_settings = {
    "parallel": False
}

def papply(df, f, catch_err=True):
    parallel = papply_settings["parallel"]
    if catch_err:
        f = catch_errors(f)
    if parallel:
        df = df.astype("object")
        ddf = dd.from_pandas(df, npartitions=ncores)
        series = ddf.apply(f, axis=1, meta=pd.Series(dtype=object)).compute(scheduler='processes')
        # res = pd.DataFrame(list(series))
        ix = ~series.isnull()
        res = pd.DataFrame(list(series[ix]), index=series[ix].index)
        # f0 = f(df.iloc[0])
        # f0 = {k: type(v) for k, v in f0.items()}
        # return df.join(x.apply(f, axis=1, result_type='expand', meta=f0).compute(scheduler='processes'))
    else:
        res = df.astype("object").apply(f, axis=1, result_type='expand')
    return df.join(res, lsuffix='_original')

def catch_errors(f, n_times=5):
    def g(*args, **kwargs):
        for _ in range(n_times - 1):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                pass
        return f(*args, **kwargs)
    return g

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 17,
    "axes.titlepad": 8,
    "figure.figsize": (6.1, 4.0),
})

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
markers = ["o", "v", "s", "p", "h", "D"]
current_markers = None
current_colors = None
def next_config():
    global current_markers, current_colors
    if current_markers is None: current_markers = markers.copy()
    if current_colors is None: current_colors = colors.copy()
    return {"marker": current_markers.pop(0), "color": current_colors.pop(0)}

def reset_config():
    global current_markers, current_colors
    current_markers = None
    current_colors = None


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


subfig_n = 0
subfig_i = 0
subfig_t_spacing = "0cm"
subfig_int_ticks = True
subfig_share = True
subfig_legend = 0
def align_figs(n, t_spacing="2.5cm", int_ticks=True, share=True, legend=2):
    global subfig_n, subfig_i, subfig_t_spacing, subfig_int_ticks, subfig_share, subfig_legend
    plt.subplots(1, n, sharey=share)
    plt.figure(figsize=(10.0, 3.8))
    subfig_n = n
    subfig_i = 0
    subfig_t_spacing = t_spacing
    subfig_int_ticks = int_ticks
    subfig_share = share
    subfig_legend = legend

def genpath(func):
    path_tmpl = "plots/" + func.__name__ + "{}{}.pdf"
    def wrapper(*args, **kwargs):
        global current_markers, current_colors, subfig_n, subfig_i
        if subfig_n > subfig_i:
            subfig_i += 1
            plt.subplot(1, subfig_n, subfig_i)
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y--%H-%M-%S-%f")
        # arg_str = [str(v) for v in args] + [f"{k}-{v}" for k, v in kwargs.items()] + [date_time]
        arg_str = [date_time]

        ax = plt.gca()
        if subfig_int_ticks:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        def savefig(postfix=None):
            postfix = "" if postfix is None else f"-{postfix}"
            path1 = path_tmpl.format("", postfix)
            path2 = path_tmpl.format("-" + "--".join(arg_str), postfix)

            if subfig_n > subfig_i: return
            if subfig_n > 0 and subfig_share:
                ax = plt.gca()
                plt.ylabel("")
                ax.tick_params(labelleft=False)
                if subfig_legend != subfig_i: ax.get_legend().remove()
                # plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(path1)
            shutil.copyfile(path1, path2)
            if is_notebook():
                plt.show()
                # print(f"saved as '{path2}'")

        x = func(*args, **kwargs, savefig=savefig)

        if subfig_n <= subfig_i:
            plt.close()
        elif subfig_share and subfig_legend != subfig_i:
            ax.get_legend().remove()
        current_markers = None
        current_colors = None
        return x
    return wrapper


class Hashable:
    def __init__(self, obj, hash):
        self._hash = hash
        self.obj = obj

    def __str__(self):
        return self._hash

    def __hash__(self):
        return self._hash


def gen_title(setup, params={}, extra_space=False):
    title = []
    trans = {
        "n": "n={}",
        "L": "L={}",
#       "tau": "$\\tau={}$",
        "t_len": "m={}",
        "n_samples": "r={}",
#       "samples_per_chain": "$m/L={}$ (\\#samples per chain)",
    }
    for k, v in setup.items():
        if len(v) == 1 and k in trans:
            title.append(trans[k].format(v[0]))
    for k, v in params.items():
        title.append(f"{k}={v}")
    enum_title = ""
    if subfig_n > 0:
        enum = {1: "(a)", 2: "(b)", 3: "(c)"}[subfig_i]
        enum_title = enum + " $\\hspace{" + subfig_t_spacing + "}$ " + ("$\\hspace{3.0cm}$" if extra_space else "")
    return enum_title + "$" + ", ".join(title) + "$"



