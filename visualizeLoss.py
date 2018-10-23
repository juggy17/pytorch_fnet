import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plotter(ds, count, patch_size, iters):
    losses = np.array(ds['loss_batch'], dtype=float)
    iterNumber = np.arange(losses.shape[0])

    filtered_losses = moving_average(losses, n=50)

    ax = plt.subplot(1, 2, count)
    plt.tight_layout()
    ax.set_title('Patch size:{:s}, {:d}k iterations'.format(patch_size, iters))
    plt.plot(iterNumber, losses, label='raw losses')
    plt.plot(iterNumber[:filtered_losses.shape[0]], filtered_losses, label='filtered losses')
    plt.grid()
    plt.legend(loc='upper right')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file1', type=str, default=None, help='CSV file containing the losses')
    parser.add_argument('--csv_file2', type=str, default=None, help='CSV file containing the losses')
    opts = parser.parse_args()

    plt.figure(1)

    ds = pd.read_csv(opts.csv_file1)
    assert('loss_batch' in ds.columns)
    plotter(ds, 1, '[32, 64, 64]', 50)

    ds = pd.read_csv(opts.csv_file2)
    assert('loss_batch' in ds.columns)
    plotter(ds, 2, '[64, 128, 128]', 20)
    plt.show()

if __name__ == "__main__":
    main()
