import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import kde
from scipy.stats import kde
from scipy import optimize
from pylab import *


# median absolute deviation
def mad(df):
    # параметр для логнормального распределения
    sigma = 1.2
    k = sp.stats.lognorm.ppf(3 / 4., s=sigma)
    median = df.median()
    return k * np.median(np.fabs(df - median))


# функция распределения температуры - сумма гауссовских
def estimation_func(x, a, b, c, d, f, g):
    return f * (1 / np.sqrt(2 * 3.14) / b * np.exp(- (x - a) ** 2 / 2 / b ** 2)) + \
           g * (1 / np.sqrt(2 * 3.14) / d * np.exp(- (x - c) ** 2 / 2 / d ** 2))


# функция макс правдоподобия для температуры
def likelihood_func(args, x):
    return -np.prod(args[4] * (1 / np.sqrt(2 * 3.14) / args[1] * np.exp(- (x - args[0]) ** 2 / 2 / args[1] ** 2)) + args[5] * (1 / np.sqrt(2 * 3.14) / args[3] * np.exp(- (x - args[2]) ** 2 / 2 / args[3] ** 2)))


# Расчет доверительных интервалов для 25%, 50% и 75% квантилей
def conf_intervals(data, qn):
    # 95% квантиль распределения Гаусса
    norm_q95 = sp.stats.norm.ppf(0.95)
    kernel = sp.stats.gaussian_kde(data)

    p25 = len(data[data < qn[5]]) / len(data)
    sigma25 = \
        (sqrt((p25 * (1 - p25)) / len(data))) / kernel(qn[5])
    p50 = len(data[data < qn[10]]) / len(data)
    sigma50 = \
        (sqrt((p50 * (1 - p50)) / len(data))) / kernel(qn[10])
    p75 = len(data[data < qn[15]]) / len(data)
    sigma75 = \
        (sqrt((p75 * (1 - p75)) / len(data))) / kernel(qn[15])

    conf_q25 = norm_q95 * sigma25
    conf_q50 = norm_q95 * sigma50
    conf_q75 = norm_q95 * sigma75
    conf = list()
    conf.append(conf_q25[0])
    conf.append(conf_q50[0])
    conf.append(conf_q75[0])

    return conf


if __name__ == "__main__":

    # step 1
    tb_3cols_all = pd.read_csv("data/data_spb.csv", index_col=0, na_values='NA',
                               usecols=['STATION', 'DATE', 'TEMP', 'STP'])
    tb_3cols = tb_3cols_all.loc[26063099999]
    # tb_3cols.TEMP = round(((tb_3cols.TEMP - 32) / 1.8), 2)  Цельсии
    tb_3cols = tb_3cols[['DATE', 'TEMP', 'STP']]
    tb_3cols.reset_index(drop=True, inplace=True)
    tb_3cols.index = tb_3cols.DATE
    tb_3cols.drop('DATE', axis=1, inplace=True)
    tb_3cols.hist()  # похоже, что у str есть пропуски

    # Тут мы указываем год для каждого показания
    tb_2cols = tb_3cols[['TEMP']]
    year_values = np.array([])
    for y in range(1990, 2021):
        m1 = '{0}-01-01'.format(y)
        m2 = '{0}-12-31'.format(y)
        srez = tb_2cols.loc[m1:m2]
        col1 = np.full(len(srez), y)
        year_values = np.hstack([year_values, col1])
    tb_2cols['YEAR'] = year_values

    # step 2a, 4, 5
    # гистограмма и непараметрическая оценка
    fig, ax = plt.subplots()
    # если хочется посмотреть по годам, можно цикл сделать
    # tb_3cols = tb_2cols[tb_2cols['YEAR'] == 2003]
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

    # step 2b
    # Вычисление выборочного среднего, дисперсии, СКО, медианы
    for y in range(1990, 2021):
        tb_3cols = tb_2cols[tb_2cols['YEAR'] == y]
        mean = tb_3cols['TEMP'].mean()
        var = tb_3cols['TEMP'].var()
        std = tb_3cols['TEMP'].std()
        median = tb_3cols['TEMP'].median()

        # Вычисление усеченного среднего, с усечением 10% наибольших и наименьших значений
        trimmed_mean = sp.stats.trim_mean(tb_3cols['TEMP'], proportiontocut=0.1)
        # Вычисление MAD-характеристики (Median Absolute Deviation)
        mad_value = mad(tb_3cols['TEMP'])
        # выборочное среднее, СКО, медиана, доверительные интервалы
        print("year = ", y)
        print(
            f'Средняя температура Фаренгейт: среднее = {int(mean)}, дисперсия = {int(var)}, СКО = {int(std)},\n'
            f'медиана = {int(median)}, усеченное среднее {int(trimmed_mean)}, MAD = {int(mad_value)}')

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

    # step 3 ящик с усами

    # tb_2cols = tb_3cols[['TEMP']]
    # tb_2cols.boxplot()
    # plt.title("Box-and-whiskers")
    # plt.show()
    tb_2box = tb_2cols[['TEMP']]
    tb_2box.index = pd.to_datetime(tb_2box.index)
    tb_2box = tb_2box.loc['1990-01-01':'2019-12-31']
    fig, ax = plt.subplots()

    groups = tb_2box.groupby(pd.Grouper(freq="1Y"))
    years = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
    years = pd.DataFrame(years)
    print(years)
    years.columns = range(1, 31)
    years.boxplot()
    plt.show()
    fig.savefig('Ящик с усами.png')

    # step 6 Расчет квантилей
    srez = tb_2cols[tb_2cols['YEAR'] == 2005]
    srez2 = tb_2cols[tb_2cols['YEAR'] == 2009]

    percs = np.linspace(0, 100, 21)
    qn_first = np.percentile(srez['TEMP'], percs)
    qn_second = np.percentile(srez2['TEMP'], percs)
    conf_first = conf_intervals(srez['TEMP'], qn_first)
    conf_second = conf_intervals(srez2['TEMP'], qn_first)

    # Построение квантильного биплота для двух случайных величин
    plt.figure(figsize=(12, 12))
    min_qn = np.min([qn_first.min(), qn_second.min()])
    max_qn = np.max([qn_first.max(), qn_second.max()])
    x = np.linspace(min_qn, max_qn)
    plt.plot(qn_first, qn_second, ls="", marker="o", markersize=6)
    plt.plot(x, x, color="k", ls="--")
    plt.xlabel('2005')
    plt.ylabel('2009')
    plt.xlim([min_qn, 80])
    plt.ylim([min_qn, 80])
    plt.grid(True)

    # Добавление доверительных интервалов на график
    plt.errorbar(
        # [25%, 50%, 75%]
        [qn_first[5], qn_first[10], qn_first[15]],
        [qn_second[5], qn_second[10], qn_second[15]],
        xerr=conf_first,
        yerr=conf_second,
        ls='none',
        capsize=3,
        elinewidth=2
    )

    plt.title('QQ-plot')
    plt.show()

    # Определение параметров логнормального распределения средней величины температуры для 2008 года
    x = np.linspace(np.min(srez['TEMP']), np.max(srez['TEMP']))

    # Параметры распределения определяются при помощи функции fit на основе метода максимального правдоподобия
    params = sp.stats.lognorm.fit(srez['TEMP'])
    pdf = sp.stats.lognorm.pdf(x, *params)

    # step 7 Расчет критерия Колмогорова-Смирнова и хи-квадрат
    ks = sp.stats.kstest(srez['TEMP'], 'lognorm', params, N=100)
    chi2 = sp.stats.chisquare(srez['TEMP'])
    print(ks)
    print(chi2)

    # Построение квантильного биплота для эмпирического и теоретического (логнормального) распределения
    # Расчет квантилей
    percs = np.linspace(0, 100, 21)
    qn_first = np.percentile(srez['TEMP'], percs)
    qn_lognorm = sp.stats.lognorm.ppf(percs / 100.0, *params)

    # Построение квантильного биплота
    plt.figure(figsize=(12, 12))
    plt.title('QQ-plot2')
    plt.plot(qn_first, qn_lognorm, ls="", marker="o", markersize=6)
    plt.plot(x, x, color="k", ls="--")
    plt.xlim([min_qn, 70])
    plt.ylim([min_qn, 70])
    plt.xlabel(f'Эмпирическое распределение')
    plt.ylabel('Теоретическое (логнормальное) распределение')
    plt.show()

    print(qn_first, qn_second, qn_lognorm)



