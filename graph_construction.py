import networkx as nx
import random

def mergeGraphs(graph1, graph2):
    start = len(graph2.nodes())
    for node in graph1.nodes():
        graph2.add_node(node + start)

    # Add edges from graph1 to graph2
    for edge in graph1.edges():
        graph2.add_edge(edge[0] + start, edge[1] + start)
    return graph2

def makeGraphFromFile(filename):
    with open(filename, 'r') as file:
        # Read the first line from the file.
        first_line = file.readline()

        # Split the line into parts. By default, split() uses whitespace as the delimiter.
        parts = first_line.split()

        N = int(parts[0])
        type = parts[1] # ignore for now lol
        edge_type = parts[2] # ignore for now

        line = file.readline().split()
        nodeCounts = [int(x) for x in line]
        G = nx.Graph()
        G.add_nodes_from(list(range(sum(nodeCounts))))
        nodeStart = nodeCounts
        nodeStart.append(0)
        for i in range(len(nodeCounts)):
            if i > 0:
                nodeStart[i] += nodeCounts[i-1]
        M = int(file.readline())

        for i in range(M):
            line = file.readline().split()
            n1 = int(line[0])
            n2 = int(line[1])
            p = float(line[2])
            for nx1 in range(nodeStart[n1], nodeStart[n1+1]):
                for nx2 in range(nodeStart[n2], nodeStart[n2+1]):
                    if nx1 == nx2: continue
                    rp = random.random()
                    print(rp, p)
                    if rp <= p:
                        G.add_edge(nx1, nx2)
        return G

print(makeGraphFromFile("edgeconstruct.in"))