import random

import numpy as np
import networkx as nx


def testSubmodularity(n, m):
    G = nx.erdos_renyi_graph(n, 0.3)

    M = nx.adjacency_matrix(G).toarray()
    M = M / np.sum(M, axis=0)[:, None]

    """
    TODO(make this N^3)
    """
    def prob(z, y):
        c = 0
        num_runs = 10000
        for _ in range(num_runs):
            x = z
            while x != y:
                if np.random.random() < A[x, x]: break
                x = np.random.choice(n, p=M[x])
            else:
                c += 1
        if (num_runs == 0): return 0
        return c / num_runs


    A = np.diag([random.choice([0.00001,1]) for x in range(n)])
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


    for a in range(n):
        for b in range(a + 1, n):
            lhs = 0
            rhs = 0
            for v in range(n):
                l, r = f(a, b, v)
                lhs += l
                rhs += r

    print("-"*50)
    print("LHS:", lhs)
    print("RHS:", rhs)
    print(lhs-rhs)
    if(lhs < rhs):
        print("BAD FABIAN")

n = 10
m = 5

for _ in range(100):
    testSubmodularity(n, m)

    # break



