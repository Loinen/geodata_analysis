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

cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
tb_3cols = pd.read_csv("average_temp.csv", index_col=0,
                       usecols=['Year']+cols)

precip = pd.read_table('nashville_precip.txt', index_col=0,na_values='NA',
                       delim_whitespace=True)
# histogram
_ = tb_3cols.hist(grid=False, width=1.0) # sharex = True
plt.margins(0.05, 0.05)
plt.tight_layout()
plt.show()

# histogram and kernel estimation for one month
# берем мат ожидание и дисперсию
tb_3cols.fillna(value={'Jan': tb_3cols.Jan.mean()}, inplace=True)
tb_3cols_mean = tb_3cols.mean()
print(tb_3cols_mean)
tb_3cols_var = tb_3cols.var()
print(tb_3cols_var)
# Посчитаем альфа и бета
alpha_mom = tb_3cols_mean ** 2 / tb_3cols_var
beta_mom = tb_3cols_var / tb_3cols_mean

yan_avg_temp = tb_3cols['Jan'].tolist()
plt.hist(yan_avg_temp, bins='auto', density=True)

density = kde.gaussian_kde(yan_avg_temp)
# temp_grid = np.linspace(min(yan_avg_temp), max(yan_avg_temp), 10)
# plt.plot(yan_avg_temp, density(yan_avg_temp))
plt.plot(np.linspace(min(yan_avg_temp), max(yan_avg_temp)), gamma.pdf(np.linspace(min(yan_avg_temp), max(yan_avg_temp)), alpha_mom[0], beta_mom[0]))
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

plt.plot(np.linspace(0, 10), gamma.pdf(np.linspace(0, 10), palpha_mom[0], pbeta_mom[0]))
plt.show()

tb_3cols.fillna(value={'Jan': tb_3cols.Jan.mean()}, inplace=True)
tb_3cols_mean = tb_3cols.mean()
tb_3cols_var = tb_3cols.var()
# Посчитаем альфа и бета
alpha_mom = tb_3cols_mean ** 2 / tb_3cols_var
beta_mom = tb_3cols_var / tb_3cols_mean
yan_avg_temp = tb_3cols['Jan'].tolist()
density = kde.gaussian_kde(yan_avg_temp)
tb_3cols.Jan.hist(bins='auto', density=True)
plt.plot(np.linspace(min(yan_avg_temp), max(yan_avg_temp)), gamma.pdf(np.linspace(min(yan_avg_temp), max(yan_avg_temp)), alpha_mom[0], beta_mom[0]))
plt.show()
