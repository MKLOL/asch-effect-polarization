import random
import math

from utils import getGraph
from mse_graph_calculator import approximateMseFaster
from mse_stooges_resistance_greedy import greedyResistanceNegative
import numpy as np

def randResist():
    if random.random() <= 0.999:
        return 1
    return 0

def getRandNeighbour(s, N):
    stooges = list(s)
    r = random.random()
    ls = [(randResist(), x) for x in range(N)]
    random.shuffle(ls)
    if r < 0.4:
        random.shuffle(stooges)
        stooges.pop()
        stooges.append(ls[0])
    elif r < 0.8:
        random.shuffle(stooges)
        stooges.pop()
        stooges.pop()
        stooges.append(ls[0])
        stooges.append(ls[1])
    elif (r < 0.95):
        random.shuffle(stooges)
        stooges.pop()
        stooges.pop()
        stooges.pop()
        stooges.append(ls[0])
        stooges.append(ls[1])
        stooges.append(ls[2])
    else:
        return ls[:len(stooges)]
    return stooges


def f(oldV, newV, temps):
    print("a=", oldV)
    print("b=", newV)
    print("t=", temps)
    print(newV, oldV, (newV - oldV), temps, (newV - oldV) / temps)
    print(math.exp((newV - oldV) / temps))
    return math.exp((newV - oldV) / temps)


def temp(t):
    return t * 0.99999


def getResistances(stooges, r):
    ret = list(r)
    for x in stooges:
        ret[x[1]] = x[0]
    return ret


def getBestSolution(G, s, r, stooges, initTemp=0.6, steps=40000):
    curValue, x_start = approximateMseFaster(G, s, resistances=getResistances(stooges, r))
    starVal = curValue
    retVal = curValue
    retS = stooges
    ogs = list(stooges)
    while(initTemp >= 0.1):
        initTemp = temp(initTemp)
        initTemp = max(0.000001, initTemp)
        newStooges = getRandNeighbour(stooges, len(G.nodes))
        newValue, x_start = approximateMseFaster(G, s, resistances=getResistances(newStooges, r))
        fVal = f(curValue*20000, newValue*20000, initTemp)
        rnd = random.random()
        print(fVal, rnd, fVal >= rnd)
        if fVal > rnd:
            print("SWAP")
            stooges = newStooges
            curValue = newValue
        if newValue > retVal:
            print("GOOD")
            retVal = newValue
            retS = newStooges
        print(newValue, (retVal-curValue), curValue, retVal, starVal, retVal/starVal, len(ogs), len(set(stooges).intersection(ogs)))
    return retS, retVal

NRT = 20
finalList = []
for tt in range(NRT):
    G, s = getGraph("smallCommunities", n = 150)
    print("WTF")
    stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False, nrange=20)
    for l in range(len(stoogePos)):
        if len(finalList) <= l:
            finalList.append([])

        finalList[l].append(stoogePos[l][0])
    print(lsPos[-1] / lsPos[0])
    print(stoogePos)
import pickle
print(finalList)
with open("filepicketmy.txt", 'wb') as file:
    pickle.dump(finalList, file)
