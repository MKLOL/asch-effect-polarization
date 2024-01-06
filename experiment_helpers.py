import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

def record_stats(b, type=None, status=None, seed=None):
    print(f"saving {type}...")

    y = b * b
    mn = np.mean(b)
    print(mn)
    z = [abs(x - mn) for x in b]
    write_statistics_to_file(f"{status}_{seed}_{type}_stats.txt", b)

    _, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.set_title(f"SEED {seed}, Plotting opinions, {status} greedy. {type}")
    ax1.hist(b, bins=20, color='blue', edgecolor='black', range=[0, 1])

    ax2.set_title(f"SEED {seed}, Plotting x*x, {status} greedy {type}")
    ax2.hist(y, bins=20, color='blue', edgecolor='black', range=[0, 1])

    ax3.set_title(f"SEED {seed}, Plotting abs(x-avg), {status} greedy {type}")
    ax3.hist(z, bins=20, color='blue', edgecolor='black', range=[0, 1])

    plt.tight_layout()
    plt.savefig(f'plots/{status}_{seed}_{type}.pdf')
