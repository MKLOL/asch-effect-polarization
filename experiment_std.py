"""
File to test the intersection between stooges if we try to maximize / minimze.
"""
import itertools

from utils import getGraph
from mse_stooges_resistance_greedy import *


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
    return lsNeg[-1] / lsNeg[0], lsPos[-1] / lsPos[0]


def bruteForce(G, s, stoogeCount):
    allCases = list(itertools.combinations(G.nodes, stoogeCount))
    best = 0
    bst = None
    for i in range(len(allCases)):
        x = allCases[i]
        currentValue = greedyResistanceNegative(G, s, stoogeCount, positive=False, change_nodes=x)
        if currentValue[2][-1] > best:
            best = currentValue[2][-1]
            bst = currentValue
    return bst


def getRandomBrute(G, s, stoogeCount):
    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, stoogeCount, positive=False)

    print("best full greedy ratio:", lsPos[-1] / lsPos[0])
    stoogeNeg, resistNeg, lsNeg = bruteForce(G, s, stoogeCount)

    for i in range(len(lsPos)):
        print(lsPos[i] / lsPos[0], i)

    print("best random greedy ratio:", lsNeg[-1] / lsNeg[0])
    print("best random greedy DIFF:", (lsNeg[-1] / lsNeg[0]) - (lsPos[-1] / lsPos[0]))
    print("best random greedy ratio bests:", (lsNeg[-1] / lsNeg[0]) / (lsPos[-1] / lsPos[0]))
    for i in range(len(lsNeg)):
        print(lsNeg[i] / lsNeg[0], i)
    print(">>>:", lsPos[-1] - lsNeg[-1])
    return lsPos, lsNeg  # return lsNeg[-1] / lsNeg[0], lsPos[-1] / lsPos[0]
