import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
# matplotlib.use('Qt5Agg')


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

    return mse, average, median

def record_stats(b, type=None, status=None, seed=None):
    print(f"saving {type}:{status}...")

    y = b * b
    mn = np.mean(b)
    print(mn)
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