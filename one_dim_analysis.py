"""
One-dimensional data analysis. Nonparametric estimation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import re
from scipy.stats import kde
from scipy.stats.distributions import gamma
from scipy.stats import norm
from scipy.optimize import newton
from scipy.special import psi, polygamma
from scipy.stats import kde
import seaborn as sns
from math import sqrt


# Берем одну станцию, со средней температурой для каждого месяца в году
cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
tb_3cols = pd.read_csv("data/average_temp_26063099999.csv", index_col=0,
                       usecols=['Year']+cols)

print("Average temperatures in each month")
print(tb_3cols)

# непараметрические оценки - 1 - histogram
_ = tb_3cols.hist(grid=False)  # sharex=True, width=0.5
plt.title("histogram")
plt.margins(0.05, 0.05)
plt.tight_layout()
plt.show()
plt.savefig('histogram')

# берем мат ожидание и дисперсию
tb_3cols_mean = tb_3cols.mean()
print("Temperature expectations for each month")
print(tb_3cols_mean)

tb_3cols_var = tb_3cols.var()
print("Temperature variances for each month")
print(tb_3cols_var)

# непараметрические оценки - 2 - ядерные оценки
# Посчитаем альфа и бета
alpha_mom = tb_3cols_mean ** 2 / tb_3cols_var
beta_mom = tb_3cols_var / tb_3cols_mean
yan_avg_temp = tb_3cols['Apr'].tolist()
plt.hist(sorted(yan_avg_temp), bins='auto', density=True)
density = kde.gaussian_kde(sorted(yan_avg_temp))
lin = np.linspace(min(yan_avg_temp), max(yan_avg_temp))
# temp_grid = np.linspace(min(yan_avg_temp), max(yan_avg_temp), 10)
# tb_3cols.Oct.hist(bins='auto', density=True)
# plt.plot(yan_avg_temp, density(yan_avg_temp))
# plt.show()
fig, ax = plt.subplots()
plt.title("apr hist")
plt.plot(lin, gamma.pdf(lin, alpha_mom[0], beta_mom[0]))
plt.show()
fig.savefig('apr histogram')

fig, ax = plt.subplots()
axs = tb_3cols.hist(bins=15, density=True, figsize=(12, 8), sharex=True, sharey=True, grid=False)
for ax in axs.ravel():
    m = ax.get_title()
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, gamma.pdf(x, alpha_mom[m], beta_mom[m]))
    label = 'alpha = {0:.2f}\nbeta = {1:.2f}'.format(alpha_mom[m], beta_mom[m])
    ax.annotate(label, xy=(10, 0.2))
plt.show()
fig.savefig('gamma histogram')

# пуассон
plt.title("Пуассон")
y = np.random.poisson(5, size=100)
plt.hist(y, bins=12)
plt.xlabel('y')
plt.ylabel('Pr(y)')
plt.show()

poisson_like = lambda x, lam: np.exp(-lam) * (lam**x) / (np.arange(x)+1).prod()

lambda5 = np.linspace(0, 15)
x = 5
plt.title("Пуассон")
plt.plot(lambda5, [poisson_like(x, l) for l in lambda5])
plt.xlabel('$\lambda$')
plt.ylabel('L($\lambda$|x={0}'.format(x))
plt.show()
# выше макс правдоподобие при л = 5
# строим плотность для л = 5

lam = 5
xvals = np.arange(15)
plt.title("Правдоподобие?")
plt.bar(xvals, [poisson_like(x, lam) for x in xvals])
plt.xlabel('x')
plt.ylabel('Pr(X|($\lambda$|=5')
plt.show()

# Метод Ньютона-Рапсона
# func = lambda x: 3./(1 + 400*np.exp(-2*x)) - 1
# xvals = np.linspace(0, 6)
# plt.plot(xvals, func(xvals))
# plt.text(5.3, 2.1, '$f(x)$', fontsize=16)
# # zero line
# plt.plot([0,6], [0,0], 'k-')
# # value at step n
# plt.plot([4,4], [0,func(4)], 'k:')
# plt.text(4, -.2, '$x_n$', fontsize=16)
# # tangent line
# tanline = lambda x: -0.858 + 0.626*x
# plt.plot(xvals, tanline(xvals), 'r--')
# # point at step n+1
# xprime = 0.858/0.626
# plt.title("Метод Ньютона-Рапсона")
# plt.plot([xprime, xprime], [tanline(xprime), func(xprime)], 'k:')
# plt.text(xprime+.1, -.2, '$x_{n+1}$', fontsize=16)
# plt.show()

dlgamma = lambda m, log_mean, mean_log: np.log(m) - psi(m) - log_mean + mean_log
dl2gamma = lambda m, *args: 1./m - polygamma(1, m)

log_mean = tb_3cols.mean().apply(np.log)
mean_log = tb_3cols.apply(np.log).mean()

print(dlgamma, dl2gamma)
print(log_mean, mean_log)

# alpha_mle = newton(dlgamma, 2, dl2gamma, args=(log_mean[-2], mean_log[-2]))
# beta_mle = alpha_mle/tb_3cols.mean()[-2]
#
# dec = tb_3cols.Nov
# dec.hist(bins=10, grid=False, density=True)
# fig, ax = plt.subplots()
# x = np.linspace(0, dec.max())
# plt.title("Метод Ньютона-Рапсона")
# plt.plot(x, gamma.pdf(x, alpha_mom[-2], beta_mom[-2]), 'm-')
# plt.plot(x, gamma.pdf(x, alpha_mle, beta_mle), 'r--')
# plt.show()
# plt.savefig('Newton-Raphson')

gamma.fit(tb_3cols.Nov)

# непарам. Ядерные оценки

y = np.random.random(15) * 10
x = np.linspace(0, 10, 100)
# Smoothing parameter
s = 0.4
# Calculate the kernels
kernels = np.transpose([norm.pdf(x, yi, s) for yi in y])
fig, ax = plt.subplots()

plt.title("Calculate the kernels")
plt.plot(x, kernels, 'k:')
plt.plot(x, kernels.sum(1))
plt.plot(y, np.zeros(len(y)), 'ro', ms=10)
plt.show()
fig.savefig('kernels')

fig, ax = plt.subplots()
x1 = np.random.normal(0, 3, 50)
x2 = np.random.normal(4, 1, 50)
x = np.r_[x1, x2]
plt.hist(x, bins=8)
plt.show()
fig.savefig('kernels2')

fig, ax = plt.subplots()
density = kde.gaussian_kde(x)
xgrid = np.linspace(x.min(), x.max(), 100)
plt.hist(x, bins=8, density=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.show()
fig.savefig('kernels3')


# по второму примеру - доверительные интервалы, выборочные статистики

tb_3cols_all = pd.read_csv("data/data_spb.csv", index_col=0, na_values='NA',
                       usecols=['STATION', 'DATE', 'TEMP', 'WDSP'])
tb_3cols = tb_3cols_all.loc[26063099999]

# plt.figure(figsize=(10, 8))
# # указываем X и Y
# print(tb_3cols['TEMP'])
# print(tb_3cols['DATE'])
# plt.title("Дата и температура по станции 26063099999")
# plt.scatter(tb_3cols['DATE'], tb_3cols['TEMP'])
# plt.xticks(rotation=45)
# plt.xlabel(u'Дата', fontsize=20)
# plt.ylabel(u'Температура (Фаренгейты)', fontsize=20)
# plt.legend()
# plt.show()

# Вычисление выборочного среднего, дисперсии, СКО, медианы
mean = tb_3cols['TEMP'].mean()
var = tb_3cols['TEMP'].var()
std = tb_3cols['TEMP'].std()
median = tb_3cols['TEMP'].median()

# Вычисление усеченного среднего, с усечением 10% наибольших и наименьших значений
trimmed_mean = sp.stats.trim_mean(tb_3cols['TEMP'], proportiontocut=0.1)


# median absolute deviation
def mad(df):
    # параметр для логнормального распределения
    sigma = 1.2
    k = sp.stats.lognorm.ppf(3 / 4., s=sigma)
    median = df.median()
    return k * np.median(np.fabs(df - median))


# Вычисление MAD-характеристики (Median Absolute Deviation)
mad_value = mad(tb_3cols['TEMP'])
# выборочное среднее, СКО, медиана, доверительные интервалы
print(f'Средняя температура Фаренгейт (за все годы, что не очень полезно): среднее = {int(mean)}, дисперсия = {int(var)}, СКО = {int(std)},\n'
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

# Построение гистограммы и ядерной оценки плотности
plt.figure(figsize=(10, 8))
plt.title("Построение гистограммы и ядерной оценки плотности")

kernel = sp.stats.gaussian_kde(tb_3cols['TEMP'])
min_amount, max_amount = tb_3cols['TEMP'].min(), tb_3cols['TEMP'].max()
x = np.linspace(min_amount, max_amount, len(tb_3cols))
kde_values = kernel(x)

den = tb_3cols['TEMP'].tolist()
density = kde.gaussian_kde(sorted(den))
# norm_hist=True,
sns.displot(tb_3cols['TEMP'], kde=False,  label=f'Средняя температура')
plt.plot(x, kde_values)
plt.show()

# fig, ax = plt.subplots()
# plt.title("Построение гистограммы и ядерной оценки плотности 2")
# plt.ylabel('p')
# plt.xlabel('Средняя температура')
# # Отображаем значения по оси абсцисс только в интервале [0, 10000]
# plt.xlim(0, 10e4)
# plt.legend()
# plt.show()
# fig.savefig('ядерной оценки плотности')

tb_2cols = tb_3cols[['DATE', 'TEMP']]
tb_2cols.reset_index(drop=True, inplace=True)
tb_2cols.index = tb_2cols.DATE
tb_2cols.drop('DATE', axis=1, inplace=True)
tb_2cols.hist()
plt.title("Гистограмма временного ряда loc[26063099999]")
plt.xlabel(u'Дата', fontsize=20)
plt.ylabel(u'Температура (Фаренгейты)', fontsize=20)
plt.show()

fig = plt.figure()
plt.title("График распределения с использованием функции сглаживания")
tb_2cols.plot.kde()
plt.xlabel(u'Дата', fontsize=20)
plt.ylabel(u'Температура (Фаренгейты)', fontsize=20)
plt.show()
fig.savefig('График распределения с использованием функции сглаживания.png')

# ящик с усами
fig, ax = plt.subplots()
srez = tb_2cols.loc['2000-01-01':'2000-12-31']
srez.index = pd.to_datetime(srez.index)
print(srez)
groups = srez.groupby(pd.Grouper(freq="1M"))
print(srez)
month = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
month = pd.DataFrame(month)
month.columns = range(1, 13)
month.boxplot()
plt.show()
fig.savefig('Ящик с усами.png')

# задача - сравнить два распределения. Возьмем за 2000 год и 2010 год (можно ветер с темп, но смысл?)
srez2 = tb_2cols.loc['2010-01-01':'2010-12-31']

print(f'Количество дней в 2010 году: {len(srez2)}')
print(f'Количество дней в 2000 году: {len(srez)}')

col1 = np.full(len(srez), 2000)
col2 = np.full(len(srez2), 2010)
srez['TIME'] = col1
srez2['TIME'] = col2
sr_2years = pd.concat([srez, srez2], ignore_index=True)
fig, ax = plt.subplots()
plt.figure(figsize=(8, 8))
sns.displot(data=sr_2years, kde=True, x='TEMP', color='green', hue="TIME")
# sns.displot(data=srez, kde=True, x='TEMP', color='green')
# sns.displot(data=srez2, kde=True, x='TEMP', color='red')
plt.ylabel('p')
plt.xlabel('Температура')
plt.xlim(0, 90)
plt.show()
fig.savefig('2000-2010.png')  # не сохраняется

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

    return [conf_q25, conf_q50, conf_q75]

# Расчет квантилей
percs = np.linspace(0, 100, 21)
qn_first = np.percentile(srez['TEMP'], percs)
qn_second = np.percentile(srez2['TEMP'], percs)

conf_first = conf_intervals(srez['TEMP'], qn_first)
conf_second = conf_intervals(srez2['TEMP'], qn_first)

print(f'25%, 50%, 75% - доверительные интервалы для температуры за 2000 год:\n {conf_first}')
print(f'25%, 50%, 75% - доверительные интервалы для температуры за 2010 год:\n {conf_second}')

#%%

# Построение квантильного биплота для двух случайных величин
plt.figure(figsize=(12, 12))

min_qn = np.min([qn_first.min(), qn_second.min()])
max_qn = np.max([qn_first.max(), qn_second.max()])
x = np.linspace(min_qn, max_qn)

plt.plot(qn_first, qn_second, ls="", marker="o", markersize=6)
plt.plot(x, x, color="k", ls="--")
plt.xlabel('2000')
plt.ylabel('2010')
plt.xlim([min_qn, 100])
plt.ylim([min_qn, 100])
plt.grid(True)
conf1 = list()
conf1.append(conf_first[0][0])
conf1.append(conf_first[1][0])
conf1.append(conf_first[2][0])
conf2 = list()
conf2.append(conf_second[0][0])
conf2.append(conf_second[1][0])
conf2.append(conf_second[2][0])
print('rr', conf1, conf2)
# Добавление доверительных интервалов на график
plt.errorbar(
    # [25%, 50%, 75%]
    [qn_first[5], qn_first[10], qn_first[15]],
    [qn_second[5], qn_second[10], qn_second[15]],
    xerr=conf1,
    yerr=conf2,
    ls='none',
    capsize=3,
    elinewidth=2
)

plt.title('QQ-plot')
plt.show()

# Определение параметров логнормального распределения средней величины температуры 2000
x = np.linspace(np.min(srez['TIME']), np.max(srez['TIME']))

# Параметры распределения определяются при помощи функции fit на основе метода максимального правдоподобия
params = sp.stats.lognorm.fit(srez['TIME'])
pdf = sp.stats.lognorm.pdf(x, *params)

# Расчет критерия Колмогорова-Смирнова и хи-квадрат
ks = sp.stats.kstest(srez['TIME'], 'lognorm', params, N=100)
chi2 = sp.stats.chisquare(srez['TIME'])
print(ks)
print(chi2)

# Построение квантильного биплота для эмпирического и теоретического (логнормального) распределения

# Расчет квантилей
percs = np.linspace(0, 100, 21)
qn_first = np.percentile(srez['TIME'], percs)
qn_lognorm = sp.stats.lognorm.ppf(percs / 100.0, *params)

# Построение квантильного биплота
plt.figure(figsize=(10, 10))
plt.plot(qn_first, qn_lognorm, ls="", marker="o", markersize=6)
plt.plot(x, x, color="k", ls="--")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel(f'Эмпирическое распределение')
plt.ylabel('Теоретическое (логнормальное) распределение')
plt.show()


