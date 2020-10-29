"""
One-dimensional data analysis. Nonparametric estimation
"""
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.stats import kde
from scipy.stats.distributions import gamma
from scipy.stats import norm
from scipy.optimize import newton
from scipy.special import psi, polygamma
from scipy.stats import kde
import seaborn as sns


cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
tb_3cols = pd.read_csv("average_temp.csv", index_col=0,
                       usecols=['Year']+cols)
print(tb_3cols)
precip = pd.read_table('nashville_precip.txt', index_col=0, na_values='NA',
                       delim_whitespace=True)
# histogram
_ = tb_3cols.hist(grid=False)  # sharex=True, width=0.5
plt.margins(0.05, 0.05)
plt.tight_layout()
plt.show()
plt.savefig('histogram')

# histogram and kernel estimation for one month
# берем мат ожидание и дисперсию
tb_3cols_mean = tb_3cols.mean()
print(tb_3cols_mean)
tb_3cols_var = tb_3cols.var()
print(tb_3cols_var)
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
plt.plot(lin, gamma.pdf(lin, alpha_mom[0], beta_mom[0]))
plt.show()
plt.savefig('apr histogram')

axs = tb_3cols.hist(bins=15, density=True, figsize=(12, 8), sharex=True, sharey=True, grid=False)

for ax in axs.ravel():
    m = ax.get_title()
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, gamma.pdf(x, alpha_mom[m], beta_mom[m]))
    label = 'alpha = {0:.2f}\nbeta = {1:.2f}'.format(alpha_mom[m], beta_mom[m])
    ax.annotate(label, xy=(10, 0.2))
plt.show()
plt.savefig('gamma histogram')


# пуассон
y = np.random.poisson(5, size=100)
plt.hist(y, bins=12)
plt.xlabel('y')
plt.ylabel('Pr(y)')
plt.show()

poisson_like = lambda x, lam: np.exp(-lam) * (lam**x) / (np.arange(x)+1).prod()

lambda5 = np.linspace(0, 15)
x = 5
plt.plot(lambda5, [poisson_like(x, l) for l in lambda5])
plt.xlabel('$\lambda$')
plt.ylabel('L($\lambda$|x={0}'.format(x))
plt.show()
# выше макс правдоподобие при л = 5
# строим плотность для л = 5

lam = 5
xvals = np.arange(15)
plt.bar(xvals, [poisson_like(x, lam) for x in xvals])
plt.xlabel('x')
plt.ylabel('Pr(X|($\lambda$|=5')
plt.show()

# Метод Ньютона-Рапсона

func = lambda x: 3./(1 + 400*np.exp(-2*x)) - 1
xvals = np.linspace(0, 6)
plt.plot(xvals, func(xvals))
plt.text(5.3, 2.1, '$f(x)$', fontsize=16)
# zero line
plt.plot([0,6], [0,0], 'k-')
# value at step n
plt.plot([4,4], [0,func(4)], 'k:')
plt.text(4, -.2, '$x_n$', fontsize=16)
# tangent line
tanline = lambda x: -0.858 + 0.626*x
plt.plot(xvals, tanline(xvals), 'r--')
# point at step n+1
xprime = 0.858/0.626
plt.plot([xprime, xprime], [tanline(xprime), func(xprime)], 'k:')
plt.text(xprime+.1, -.2, '$x_{n+1}$', fontsize=16)
plt.show()
plt.savefig('Newton1')


dlgamma = lambda m, log_mean, mean_log: np.log(m) - psi(m) - log_mean + mean_log
dl2gamma = lambda m, *args: 1./m - polygamma(1, m)

log_mean = tb_3cols.mean().apply(np.log)
mean_log = tb_3cols.apply(np.log).mean()

print(dlgamma, dl2gamma)
print(log_mean, mean_log)
alpha_mle = newton(dlgamma, 2, dl2gamma, args=(log_mean[-2], mean_log[-2]))
beta_mle = alpha_mle/tb_3cols.mean()[-2]

dec = tb_3cols.Nov
dec.hist(bins=10, grid=False, density=True)
x = np.linspace(0, dec.max())
plt.plot(x, gamma.pdf(x, alpha_mom[-2], beta_mom[-2]), 'm-')
plt.plot(x, gamma.pdf(x, alpha_mle, beta_mle), 'r--')
plt.show()
plt.savefig('Newton-Raphson')

gamma.fit(tb_3cols.Nov)

# непарам. Ядерные оценки

y = np.random.random(15) * 10
x = np.linspace(0, 10, 100)
# Smoothing parameter
s = 0.4
# Calculate the kernels
kernels = np.transpose([norm.pdf(x, yi, s) for yi in y])
plt.plot(x, kernels, 'k:')
plt.plot(x, kernels.sum(1))
plt.plot(y, np.zeros(len(y)), 'ro', ms=10)
plt.show()
plt.savefig('kernels')

x1 = np.random.normal(0, 3, 50)
x2 = np.random.normal(4, 1, 50)
x = np.r_[x1, x2]
plt.hist(x, bins=8)
plt.show()
plt.savefig('kernels2')

density = kde.gaussian_kde(x)
xgrid = np.linspace(x.min(), x.max(), 100)
plt.hist(x, bins=8, density=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.show()
plt.savefig('kernels3')


# по второму примеру - доверительные интервалы, выборочные статистики

# tb_3cols = pd.read_csv("data/data_one_dim.csv", index_col=1, na_values='NA',
#                        usecols=['STATION', 'DATE', 'TEMP', 'WDSP'])
# plt.figure(figsize=(10, 8))
# #указываем X и Y
# plt.scatter(tb_3cols['DATE'], tb_3cols['amount'])
# plt.xticks(rotation=45)
# plt.xlabel(u'Номер клиента', fontsize = 20)
# plt.ylabel(u'Средняя транзакция', fontsize = 20)

