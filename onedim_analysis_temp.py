import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import kde
from scipy import optimize
from pylab import *
import random


# функция распределения температуры - сумма гауссовских
def estimation_func(x, a, b, c, d, f, g):
    return f * (1 / np.sqrt(2 * 3.14) / b * np.exp(- (x - a) ** 2 / 2 / b ** 2)) + \
           g * (1 / np.sqrt(2 * 3.14) / d * np.exp(- (x - c) ** 2 / 2 / d ** 2))

# функция макс правдоподобия для температуры
def likelihood_func(args, x):
    return -np.prod(args[4] * (1 / np.sqrt(2 * 3.14) / args[1] * np.exp(- (x - args[0]) ** 2 / \
                    2 / args[1] ** 2)) + args[5] * (1 / np.sqrt(2 * 3.14) / args[3] * \
                    np.exp(- (x - args[2]) ** 2 / 2 / args[3] ** 2)))


if __name__ == "__main__":
    # step 1
    tb_3cols_all = pd.read_csv("data/data_spb.csv", index_col=0, na_values='NA',
                               usecols=['STATION', 'DATE', 'TEMP'])

    tb_3cols = tb_3cols_all.loc[26063099999]
    # # убираем станцию
    # tb_3cols.reset_index(drop=True, inplace=True)
    # tb_3cols.index = tb_3cols.DATE
    # tb_3cols.drop('DATE', axis=1, inplace=True)
    # pd.to_datetime(tb_3cols.index)

    # step 2a, 4, 5
    # гистограмма и непараметрическая оценка
    fig, ax = plt.subplots()

    y, x, _ = hist(tb_3cols['TEMP'].tolist(), 100, alpha=.3, label='data', density=True)

    density = kde.gaussian_kde(sorted(tb_3cols['TEMP']))
    kernel = sp.stats.gaussian_kde(tb_3cols['TEMP'])
    temp_grid = np.linspace(min(tb_3cols['TEMP']), max(tb_3cols['TEMP']), 100)
    plt.plot(temp_grid, density(temp_grid), label="Kernel estimation")

    x = (x[1:] + x[:-1]) / 2  # now x and y have the same size

    args_0 = (35, 10, 60, 10, 1, 1)
    params, cov = optimize.curve_fit(estimation_func, x, y, args_0)

    plt.plot(x, estimation_func(x, *params), color='green', lw=3, label='Least squares estimation')

    args_0 = params
    lik_model = optimize.minimize(likelihood_func, args_0, args=tb_3cols['TEMP'].tolist())
    plt.plot(temp_grid, estimation_func(temp_grid, *lik_model.x), color='red', label='MLE')
    legend()

    plt.legend()
    plt.show()

    # step 6
    percs = np.linspace(0, 100, 21)

    # сэмплирование рандомных значений из распределения
    dist = estimation_func(x, *lik_model.x)
    dist = dist / dist.sum()
    est_sample = np.random.choice(x, p=dist, size=500, replace=True)
    print(f"сэмлирование значений из распределения температуры\n", est_sample)

    plt.plot(temp_grid, estimation_func(temp_grid, *lik_model.x), color='red', label='MLE')
    plt.plot(temp_grid, density(temp_grid), label="Kernel estimation")
    density = kde.gaussian_kde(sorted(est_sample))
    grid = np.linspace(min(est_sample), max(est_sample), 100)
    plt.plot(grid, density(grid), label="Sample")

    plt.legend()
    plt.show()

    qn_real = np.percentile(tb_3cols['TEMP'], percs)
    qn_estim = np.percentile(est_sample, percs)

    min_qn = np.min([qn_real.min(), qn_estim.min()])
    max_qn = np.max([qn_real.max(), qn_estim.max()])
    x = np.linspace(min_qn, max_qn)

    plt.plot(qn_real, qn_estim, ls="", marker="o", markersize=6)
    plt.plot(x, x, color="k", ls="--")
    plt.show()

    # step 7
    temp_sample = random.choices(tb_3cols['TEMP'].tolist(), k=200)
    ks = sp.stats.kstest(temp_sample, est_sample)
    print(ks)

    pearson = sp.stats.pearsonr(temp_sample, est_sample)
    print(f"Критерий Пирсона {pearson}")

    # step 2b
    # Вычисление выборочного среднего, дисперсии, СКО, медианы
    mean = tb_3cols['TEMP'].mean()
    var = tb_3cols['TEMP'].var()
    std = tb_3cols['TEMP'].std()
    median = tb_3cols['TEMP'].median()

    # Вычисление усеченного среднего, с усечением 10% наибольших и наименьших значений
    trimmed_mean = sp.stats.trim_mean(tb_3cols['TEMP'], proportiontocut=0.1)

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
    print("Выборочное среднее: %0.3f +/- %0.3f" % (mean, mean_conf))
    print("95%% Доверительный интервал выборочной дисперсии : (%0.3f; %0.3f)"
          % (var_conf_left, var_conf_right))
    print("95%% Доверительный интервал выборочного СКО: (%0.3f; %0.3f)"
          % (std_conf_left, std_conf_right))

    # step 3
    tb_2cols = tb_3cols[['TEMP']]
    tb_2cols.boxplot()
    plt.title("Temperature Box-and-whiskers")
    plt.show()


