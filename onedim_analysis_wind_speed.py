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
    percs = np.linspace(0, 99, 21)
    qn_real = np.percentile(tb_3cols['WDSP'], percs)
    qn_gamma = sp.stats.gamma.ppf(percs / 100.0, *params)

    min_qn = np.min([qn_real.min(), qn_gamma.min()])
    max_qn = np.max([qn_real.max(), qn_gamma.max()])
    x = np.linspace(min_qn, max_qn)

    plt.plot(x, x, color="k", ls="--")
    plt.plot(qn_real, qn_gamma, ls="", marker="o", markersize=6)
    plt.xlabel('Эмпирическое распределение')
    plt.ylabel('Гамма распределение')
    plt.show()

    # step 7
    ks = sp.stats.kstest(tb_3cols['WDSP'], 'gamma', params)
    print(ks)

    wind_sample = np.random.choice(tb_3cols['WDSP'].tolist(), size=3000, replace=True)

    # сэмплирование рандомных значений из распределения
    dist = gamma.pdf(x, *params)
    dist = dist / dist.sum()
    est_sample = np.random.choice(x, p=dist, size=3000, replace=True)

    plt.plot(x, gamma.pdf(x, *params), 'm', lw=3, label="MLE")
    plt.plot(x, density(x), label="Kernel estimation")
    density = kde.gaussian_kde(sorted(est_sample))
    grid = np.linspace(min(est_sample), max(est_sample), 100)
    plt.plot(grid, density(grid), label="Sample")

    plt.legend()
    plt.show()

    es_test = sp.stats.epps_singleton_2samp(wind_sample, est_sample)
    print(es_test)

    # step 2b
    # Вычисление выборочного среднего, дисперсии, СКО, медианы
    mean = tb_3cols['WDSP'].mean()
    var = tb_3cols['WDSP'].var()
    std = tb_3cols['WDSP'].std()
    median = tb_3cols['WDSP'].median()

    # Расчет 95% доверительного интервала для выборочного среднего
    norm_q95 = sp.stats.norm.ppf(0.95)
    mean_conf = norm_q95 * std / np.sqrt(len(tb_3cols))

    # Расчет 95% доверительных интервалов для дисперсии и СКО
    chi2_q95_left = sp.stats.chi2.ppf((1 - 0.05 / 2.0), df=len(tb_3cols) - 1)
    chi2_q95_right = sp.stats.chi2.ppf(0.05 / 2.0, df=len(tb_3cols) - 1)

    var_conf_left = var * (len(tb_3cols) - 1) / chi2_q95_left
    var_conf_right = var * (len(tb_3cols) - 1) / chi2_q95_right
    std_conf_left = np.sqrt(var_conf_left)
    std_conf_right = np.sqrt(var_conf_right)

    # Вывод полученных значений
    print("95%% Доверительный интервал выборочного среднего: (%0.3f; %0.3f)"
          % (mean - mean_conf, mean + mean_conf))
    print("95%% Доверительный интервал выборочной дисперсии : (%0.3f; %0.3f)"
          % (var_conf_left, var_conf_right))
    print("95%% Доверительный интервал выборочного СКО: (%0.3f; %0.3f)"
          % (std_conf_left, std_conf_right))

    # step 3
    tb_2cols = tb_3cols[['DATE', 'WDSP']]
    tb_2cols.boxplot()
    plt.title("Box-and-whiskers")
    plt.show()
