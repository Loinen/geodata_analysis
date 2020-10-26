"""
One-dimensional data analysis. Nonparametric estimation
"""
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.stats import kde

cols = ['Jan', 'Feb', 'Mer', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
tb_3cols = pd.read_csv("average_temp.csv", index_col=0,
                       usecols=['Year']+cols)

# histogram
_ = tb_3cols.hist(grid=False)
plt.tight_layout()
plt.show()

# histogram and kernel estimation for one month
yan_avg_temp = tb_3cols['Jan'].unique().tolist()
plt.hist(yan_avg_temp, bins='auto', density=True)

density = kde.gaussian_kde(yan_avg_temp)
temp_grid = np.linspace(min(yan_avg_temp), max(yan_avg_temp), 10)
plt.plot(yan_avg_temp, density(yan_avg_temp))
plt.show()

