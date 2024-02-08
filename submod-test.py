import numpy as np
import networkx as nx


n = 10
m = 5


for _ in range(100):
    G = nx.barabasi_albert_graph(n, m)

    M = nx.adjacency_matrix(G).toarray()
    M = M / np.sum(M, axis=0)[:, None]

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
        return c / num_runs

    # import pdb; pdb.set_trace()

    A = np.diag([0.5] * n)
    I = np.eye(n)
    P2 = np.linalg.pinv(I - (I - A) @ M) @ A
    P3 = np.linalg.pinv(I - (I - A) @ M) @ (I - A) @ M
    P = np.array([[prob(i, j) for j in range(n)] for i in range(n)])
    # print(P)

    def f(a, b, v):
        Q = 1 - P
        p = P[v,a]*P[a,b] + P[v,b]*P[b,a]
        f0 = 0
        fa = P[v,a]*Q[a,b] + p
        fb = P[v,b]*Q[b,a] + p
        fab = P[v,a]*Q[a,b] + P[v,b]*Q[b,a] + p
        return fa + fb, fab + f0


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
    assert(lhs >= rhs)

    # break



