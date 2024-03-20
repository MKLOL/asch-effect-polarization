"""
File to test the intersection between stooges if we try to maximize / minimze.
"""
import itertools

from graph_creator import getGraph
from mse_stooges_resistance_greedy import *
import numpy as np

def makeData(lsPos, lsNeg):
    print("best full greedy ratio:", lsPos[-1] / lsPos[0])
    ls1 = []
    for i in range(len(lsPos)):
        ls1.append(lsPos[i] / lsPos[0])
        print(lsPos[i] / lsPos[0], i)

    print("best random greedy ratio:", lsNeg[-1] / lsNeg[0])

    ls2 = []
    for i in range(len(lsNeg)):
        ls2.append(lsNeg[i] / lsNeg[0])
        print(lsNeg[i] / lsNeg[0], i)
    return lsNeg[-1]/lsNeg[0], lsPos[-1] / lsPos[0]

def brut(G, s, stoogeCount):
    allCases = list(itertools.combinations(G.nodes, stoogeCount))
    best = 0
    bst = None
    for i in range(len(allCases)):
        if (i % 1000 == 0):
            print("SOVING FOR i", i, len(allCases))
        x = allCases[i]
        ax = greedyResistanceNegative(G, s, stoogeCount, positive=False, change_nodes=x)
        if ax[2][-1] > best:
            best = ax[2][-1]
            bst = ax

        if (i % 1000 == 0):
            print(bst[2][-1] / bst[2][0])
    return bst
def getRandomBrut(G, s, stoogeCount):
    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, stoogeCount, positive=False)

    print("best full greedy ratio:", lsPos[-1] / lsPos[0])
    stoogeNeg, resistNeg, lsNeg = brut(G, s, stoogeCount)

    for i in range(len(lsPos)):
        print(lsPos[i] / lsPos[0], i)

    print("best random greedy ratio:", lsNeg[-1] / lsNeg[0])

    print("best random greedy DIFF:", (lsNeg[-1] / lsNeg[0]) - (lsPos[-1] / lsPos[0]))
    print("best random greedy ratio bests:", (lsNeg[-1] / lsNeg[0]) / (lsPos[-1] / lsPos[0]))
    for i in range(len(lsNeg)):
        print(lsNeg[i] / lsNeg[0], i)
    return lsNeg[-1] / lsNeg[0], lsPos[-1] / lsPos[0]


s1 = []
s2 = []
for i in range(5):
    G, s = getGraph("GNP")
    l1, l2 = getRandomBrut(G, s, int(5))
    s1.append(l1)
    s2.append(l2)

print(np.average(s1), np.average(s2), np.std(s1), np.std(s2))
