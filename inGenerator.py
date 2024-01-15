"""
Big N is how many nodes are in the Gnp
Small N is how many nodes in the smaller communities
C is how many communities
p1 is internal probability for big GNP
p2 is internal probability for small communities
p3 is probability between a big and small community.
"""

def printBigGNPSmallCommunities(BigN, smallN, C, p1, p2, p3):
    print(1 + C, "p", "e")
    print(BigN, end=" ")
    for x in range(C):
        print(smallN, end=" ")
    print("")
    for x in range(C + 1):
        print("0.5 0.1 ", end="")
    print()
    ls = []
    ls.append([0, 0, p1])
    for i in range(C):
        ls.append([i + 1, i + 1, p2])
        ls.append([i + 1, 0, p3])
    print(len(ls))
    for x in ls:
        print(*x)


printBigGNPSmallCommunities(200, 10, 10, 0.5, 0.5, 0.3)
