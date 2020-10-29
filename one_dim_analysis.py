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

# histogram and kernel estimation for one month
# берем мат ожидание и дисперсию
tb_3cols_mean = tb_3cols.mean()
print(tb_3cols_mean)
tb_3cols_var = tb_3cols.var()
print(tb_3cols_var)
# Посчитаем альфа и бета
alpha_mom = tb_3cols_mean ** 2 / tb_3cols_var
beta_mom = tb_3cols_var / tb_3cols_mean

yan_avg_temp = tb_3cols['Jul'].tolist()
plt.hist(yan_avg_temp, bins='auto', density=True)

density = kde.gaussian_kde(yan_avg_temp)
# temp_grid = np.linspace(min(yan_avg_temp), max(yan_avg_temp), 10)
# plt.plot(yan_avg_temp, density(yan_avg_temp))
lin = np.linspace(min(yan_avg_temp), max(yan_avg_temp))
plt.plot(lin, gamma.pdf(lin, alpha_mom[0], beta_mom[0]))
plt.show()

# берем мат ожидание и дисперсию
precip.fillna(value={'Oct': precip.Oct.mean()}, inplace=True)

precip_mean = precip.mean()
precip_var = precip.var()
# Посчитаем альфа и бета
palpha_mom = precip_mean ** 2 / precip_var
pbeta_mom = precip_var / precip_mean
yan_avg_temp = precip['Jan'].tolist()
density = kde.gaussian_kde(yan_avg_temp)
precip.Jan.hist(bins=20, density=True)
lin = np.linspace(min(yan_avg_temp), max(yan_avg_temp))
plt.plot(lin, gamma.pdf(lin, palpha_mom[0], pbeta_mom[0]))
plt.show()

axs = precip.hist(bins=15, density=True, figsize=(12, 8), sharex=True, sharey=True, grid=False)

for ax in axs.ravel():
    m = ax.get_title()
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, gamma.pdf(x, palpha_mom[m], pbeta_mom[m]))
    label = 'alpha = {0:.2f}\nbeta = {1:.2f}'.format(palpha_mom[m],pbeta_mom[m])
    ax.annotate(label, xy=(10, 0.2))
plt.show()

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

dlgamma = lambda m, log_mean, mean_log: np.log(m) - psi(m) - log_mean + mean_log
dl2gamma = lambda m, *args: 1./m - polygamma(1, m)

log_mean = precip.mean().apply(np.log)
mean_log = precip.apply(np.log).mean()

alpha_mle = newton(dlgamma, 2, dl2gamma, args=(log_mean[-1], mean_log[-1]))
beta_mle = alpha_mle/precip.mean()[-1]

dec = precip.Dec
dec.hist(bins=10, grid=False, density=True)
x = np.linspace(0, dec.max())
plt.plot(x, gamma.pdf(x, palpha_mom[-1], pbeta_mom[-1]), 'm-')
plt.plot(x, gamma.pdf(x, alpha_mle, beta_mle), 'r--')
plt.show()

gamma.fit(precip.Dec)

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

x1 = np.random.normal(0, 3, 50)
x2 = np.random.normal(4, 1, 50)
x = np.r_[x1, x2]
plt.hist(x, bins=8)
plt.show()

density = kde.gaussian_kde(x)
xgrid = np.linspace(x.min(), x.max(), 100)
plt.hist(x, bins=8, density=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.show()

# tb_3cols_mean = tb_3cols.mean()
# tb_3cols_var = tb_3cols.var()
# # Посчитаем альфа и бета
# alpha_mom = tb_3cols_mean ** 2 / tb_3cols_var
# beta_mom = tb_3cols_var / tb_3cols_mean
# yan_avg_temp = tb_3cols['Jan'].tolist()
# density = kde.gaussian_kde(yan_avg_temp)
# tb_3cols.Jan.hist(bins='auto', density=True)
# plt.plot(np.linspace(min(yan_avg_temp), max(yan_avg_temp)), gamma.pdf(np.linspace(min(yan_avg_temp), max(yan_avg_temp)), alpha_mom[0], beta_mom[0]))
# plt.show()
