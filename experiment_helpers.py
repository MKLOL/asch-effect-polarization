import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
# matplotlib.use('Qt5Agg')

def record_stats(ab, type=None, status=None, seed=None):
    print(f"saving {type}:{status}...")

    b = ab
    y = b * b
    mn = np.mean(b)
    z = [abs(x - mn) for x in b]

    try:
        os.mkdir(f"plots/{type}")
    except OSError:
        pass

    stats = write_statistics_to_file(f"plots/{type}/{status}_{seed}_stats.txt", b)

    _, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.set_title(f"SEED {seed}, Plotting opinions, {status} greedy. {type}")
    ax1.hist(b, bins=20, color='blue', edgecolor='black', range=[0, 1])

    ax2.set_title(f"SEED {seed}, Plotting x*x, {status} greedy {type}")
    ax2.hist(y, bins=20, color='blue', edgecolor='black', range=[0, 1])

    ax3.set_title(f"SEED {seed}, Plotting abs(x-avg), {status} greedy {type}")
    ax3.hist(z, bins=20, color='blue', edgecolor='black', range=[0, 1])

    plt.tight_layout()
    plt.savefig(f'plots/{type}/{status}_{seed}.pdf')

    return stats

def plot_change(stats, type):
    _, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot([x[0] for x in stats])
    ax1.set_title("MSE")
    ax2.plot([x[1] for x in stats])
    ax2.set_title("Average")
    ax3.plot([x[2] for x in stats])
    ax3.set_title("Median")
    plt.tight_layout()
    plt.savefig(f'plots/{type}/change.pdf')