import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx

def testSubmodularity(n, m):
    G = nx.random_tree(n)
    n = len(G.nodes)
    print(n)
    M = nx.adjacency_matrix(G).toarray()
    M = M / np.sum(M, axis=0)[:, None]

    """
    TODO(make this N^3)
    """
    def prob(z, y):
        c = 0
        num_runs = 20000
        for _ in range(num_runs):
            x = z
            while x != y:
                if np.random.random() < A[x, x]: break
                x = np.random.choice(n, p=M[x])
            else:
                c += 1
        if (num_runs == 0): return 0
        return c / num_runs


    A = np.diag([random.choice([0.66,1]) for _ in range(n)])
    I = np.eye(n)
    P = np.array([[prob(i, j) for j in range(n)] for i in range(n)])

    def f(a, b, v):
        Q = 1 - P
        p = P[v,a]*P[a,b] + P[v,b]*P[b,a]
        f0 = 0
        fa = P[v,a]*Q[a,b] + p
        fb = P[v,b]*Q[b,a] + p
        fab = P[v,a]*Q[a,b] + P[v,b]*Q[b,a] + p
        return fa**2 + fb**2, fab**2 + f0**2


    flhs = 0
    frhs = 0
    for v in range(n):
        lhs = 0
        rhs = 0
        for a in range(n):
            for b in range(a + 1, n):
                l, r = f(a, b, v)
                lhs += l
                rhs += r
        print(lhs, rhs)
        flhs += lhs
        frhs += rhs
        if (lhs < rhs):
            print("BAD FABIAN local")

    print("-"*50)
    print("LHS:", flhs)
    print("RHS:", frhs)
    print(flhs-frhs)
    if(flhs < frhs):
        print("BAD FABIAN global")


n = 12
m = 5

for _ in range(100):
    testSubmodularity(n, m)

    # break



