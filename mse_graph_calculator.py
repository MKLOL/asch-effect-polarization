from cython_mse_graph_calculator import *


def approximateMseFaster(*args, targetNodes=None, theta=None, **kwargs):
    x = approximate_opinions(*args, **kwargs)
    x_mse = np.var(x[targetNodes]) if theta is None else np.mean((x[targetNodes] - theta) ** 2)
    return x_mse, x


def approximateMseFast(G, slist, resistances=None, max_iterations=100, eps=1e-5, targetNodes=None):
    n = len(G.nodes)
    x = slist
    tnodes = targetNodes
    if targetNodes is None:
        tnodes = G.nodes

    for i in range(max_iterations):
        x_new = np.empty(n)
        for u in G.nodes:
            x_u = resistances[u] * slist[u]
            x_vs = 0
            for v in G.neighbors(u): x_vs += x[v]
            x_u += (1 - resistances[u]) * x_vs / len(G[u])
            x_new[u] = x_u

        norm = np.linalg.norm(x - x_new)
        norm_inf = np.max(np.abs(x - x_new))
        if norm_inf < eps: break
        x = x_new

        print(f"Iteration {i}: {np.var(x):.7f} (change={norm:.5f})")

    # print(x)
    x_mse = np.var([x[n] for n in tnodes])
    return x_mse, x


def approximateMse(G, slist, resistances=None, max_iterations=100, eps=1e-5):
    n = len(G.nodes)
    W = nx.adjacency_matrix(G).toarray()
    W = W / np.sum(W, axis=0)[:, None]

    A = np.diag([0.5] * n if resistances is None else resistances)

    x = slist

    for i in range(max_iterations):
        x_new = A @ slist + (np.eye(n) - A) @ W @ x
        norm = np.linalg.norm(x - x_new)
        norm_inf = np.max(np.abs(x - x_new))
        if norm_inf < eps: break
        x = x_new

        # print(f"Iteration {i}: {norm}")

    # print(x)
    x_mse = np.var(x)  # np.mean((x - np.mean(x))**2)
    return x_mse


def calculateMse(G, slist, numCompute=100, resistances=None, targetNodes=None):
    tnodes = targetNodes
    if tnodes is None:
        tnodes = G.nodes
    n = len(G.nodes)
    W = nx.adjacency_matrix(G).toarray()
    W = W / np.sum(W, axis=0)[:, None]

    A = np.diag([0.5] * n if resistances is None else resistances)

    I = np.eye(n)
    X = np.linalg.pinv(I - (I - A) @ W) @ A

    s_val = []
    x_val = []
    for _ in range(numCompute):
        s = None
        if slist is None:
            s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)
        else:
            s = slist
        xs = X @ s
        st = [s[x] for x in tnodes]
        xst = [xs[x] for x in tnodes]
        avgs = np.mean(st)
        avgx = np.mean(xst)
        s_val.append(np.mean([(x - avgs) ** 2 for x in st]))
        x_val.append(np.mean([(x - avgx) ** 2 for x in xst]))

    s_mse = np.mean(s_val)
    x_mse = np.mean(x_val)

    return (s_mse, x_mse)
