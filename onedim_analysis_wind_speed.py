import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import kde
from scipy import optimize
from pylab import *
import random
from scipy.stats.distributions import gamma

if __name__ == "__main__":
    # step 1
    tb_3cols_all = pd.read_csv("data/data_spb.csv", index_col=0, na_values='NA',
                               usecols=['STATION', 'DATE', 'WDSP'])
    tb_3cols = tb_3cols_all.loc[26063099999]

    # step 2a, 4, 5
    # гистограмма и непараметрическая оценка
    fig, ax = plt.subplots()

    y, x, _ = hist(tb_3cols['WDSP'].tolist(), 50, alpha=.3, label='data', density=True)

    density = kde.gaussian_kde(sorted(tb_3cols['WDSP']))
    kernel = sp.stats.gaussian_kde(tb_3cols['WDSP'])
    temp_grid = np.linspace(min(tb_3cols['WDSP']), max(tb_3cols['WDSP']), 100)
    plt.plot(temp_grid, density(temp_grid), label="Kernel estimation")

    params = gamma.fit(tb_3cols['WDSP'])
    plt.plot(x, gamma.pdf(x, *params), 'm', lw=3, label="MLE")

    x = (x[1:] + x[:-1]) / 2  # now x and y have the same size

    params_ls, cov = optimize.curve_fit(gamma.pdf, x, y, params)
    plt.plot(x, gamma.pdf(x, *params), color='y', label='Least squares estimation')

    plt.legend()
    plt.show()

    # step 6
    percs = np.linspace(0, 100, 21)
    qn_real = np.percentile(tb_3cols['WDSP'], percs)
    qn_gamma = sp.stats.lognorm.ppf(percs / 100.0, *params)