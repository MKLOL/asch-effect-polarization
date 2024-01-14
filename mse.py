import networkx
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Qt5Agg')
import graph_construction
from mse_stooges_resistance_greedy import *
import random
import numpy as np

def write_statistics_to_file(filename, data):
    # Perform computations
    data_array = np.array(data)
    mse = np.mean((data_array - np.mean(data_array)) ** 2)
    average = np.mean(data_array)
    total_sum = np.sum(data_array)
    median = np.median(data_array)
    quantiles = np.percentile(data_array, [25, 50, 75])

    # Open the file once all computations are done
    with open(filename, 'w') as file:
        file.write(f'Mean Squared Error: {mse}\n')
        file.write(f'Average: {average}\n')
        file.write(f'Sum: {total_sum}\n')
        file.write(f'Median: {median}\n')
        file.write(f'Quantiles (25th, 50th, 75th): {quantiles}\n')


"""
n = 44

G = nx.erdos_renyi_graph(n, p=0.3)
G.add_edges_from(zip(G.nodes, G.nodes))
edgeCount = 10

print(getEdgeList(G, edgeCount))"""
seed = 111
random.seed(seed)
np.random.seed(seed)
"""
Main types are: GNP, START, and anything else

Anything else signifies you're reading from an .in file. 
"""
type = "COMMUNITIES"
status = "pre"

s = []
G, s = graph_construction.makeGraphFromFile("smallCommunities.in")
# G = networkx.erdos_renyi_graph(320, 0.2)
# G = networkx.star_graph(320)


n = len(G.nodes)
target = None
"""
print(G.nodes)
target = random.sample(sorted(G.nodes), 2)
nx = list(G.neighbors(target[0]))
ny = list(G.neighbors(target[1]))

target += nx[:4]
target += ny[:4]
"""

if type == "GNP" or type == "star_random":
    s = np.clip(np.random.normal(0.5, 0.5, n), 0, 1)

G.add_edges_from(zip(G.nodes, G.nodes))
n = len(G.nodes)
if type == "STAR":
    s.append(1)
    for x in range(n-1):
        s.append(0.498433)
print(s, len(s), n)
a, b = approximateMseFaster(G, s, targetNodes=target)
y = [x * x for x in b]
mn = np.mean(b)
print(mn)
z = [abs(x - mn) for x in b]
write_statistics_to_file(f"{status}_{seed}_{type}_stats.txt", b)
plt.title(f"SEED {seed}, Plotting opinions, {status} greedy. {type}")
plt.hist(b, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_opinion.png')

plt.title(f"SEED {seed}, Plotting x*x, {status} greedy {type}")
plt.hist(y, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_square.png')


plt.title(f"SEED {seed}, Plotting abs(x-avg), {status} greedy {type}")
plt.hist(z, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_abs.png')

ls, resist = greedyResistance(G, s, 12)
print(len(G.nodes))
print("best ratio:", ls[-1]/ls[0])
for i in range(len(ls)):
    print(ls[i]/ls[0], i)

plt.title(f"SEED {seed}, Plotting stooges vs MSE: {status} greedy {type}")
plt.hist(ls, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_mse_by_time.png')



a, b = approximateMseFaster(G, s, resistances=resist, targetNodes=target)
y = [x *x for x in b]
mn = np.mean(b)
print(mn)
z = [abs(x - mn) for x in b]

status = "post"
write_statistics_to_file(f"{status}_{seed}_{type}_stats.txt", b)

plt.title(f"SEED {seed}, Plotting opinions, {status} greedy. {type}")
plt.hist(b, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_opinion.png')

plt.title(f"SEED {seed}, Plotting x*x, {status} greedy {type}")
plt.hist(y, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_square.png')

plt.title(f"SEED {seed}, Plotting abs(x-avg), {status} greedy {type}")
plt.hist(z, bins=20, color='blue', edgecolor='black', range=[0, 1])
plt.savefig(f'{status}_{seed}_{type}_abs.png')

"""
130
10,20,30 nodes
15,30,33 % increase

150
10,20,30
16,29,42

230 (160 max)
10,20,30,40,60
7,18,21,23,32


TWO CLUSTERS:
320 (160 - 160)
1.07126948861406 10
1.1394600994370023 20
1.2074788354896184 30
1.2752423318981125 40
1.3424463722417608 50
1.4004436419655881 59


star init and star final run on 320 star graph node. 
Final ration 32. 
"""