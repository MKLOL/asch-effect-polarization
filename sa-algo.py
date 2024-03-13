import random
import math

from graph_creator import getGraph
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


G, s = getGraph("GNP")
resistances = 0.5 * np.ones(len(G.nodes))
N = 20
ls = [x/N for x in range(0,N+1)]
cnt = 0
bcnt = 0
for ax in range(1000):
    mx = 0
    mi = 0
    for x in ls:
        resistances[random.randint(0, len(G.nodes))-1] = x
        curValue, x_start = approximateMseFaster(G, s, resistances)
        if (mx < curValue):
            mx = curValue
            mi = x
    if (mi == 0 or mi == 1.0):
        cnt += 1
    bcnt += 1
    print(mx, mi)
print(cnt / bcnt)
"""stoogePos, resistPos, lsPos = greedyResistanceNegative(G, s, int(math.log2(len(G.nodes)) * 5), positive=False)

print(lsPos[-1] / lsPos[0])
print(stoogePos)
newStooges, value = getBestSolution(G, s, resistances, stoogePos)

print(lsPos[-1] / lsPos[0], value / lsPos[0], value / lsPos[-1], value, lsPos[-1])
"""