import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from scipy.stats import kde
from scipy import stats
from scipy import interpolate
from scipy.stats import norm


if __name__ == "__main__":
    data = pd.read_csv("data/data_spb.csv",
                       usecols=['STATION', 'DATE', 'TEMP', 'SLP', 'WDSP'], index_col=0)
    # тут параметры с норм корреляцией(0,5) хотя бы с 1
    #                    usecols=['STATION', 'DATE', 'GUST', 'SLP',
    #                             'SNDP', 'STP', 'TEMP',  'WDSP'], index_col=0)
    data = data.loc[26063099999]

    # TEMP - Mean temperature (.1 Fahrenheit)
    # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9 (.1 mb)
    # WDSP – Mean wind speed (.1 knots)

    # удаление пропущенных значений
    data = data.replace(9999.9, np.nan, regex=True)
    data = data.dropna(subset=['SLP'])

    # data.loc[data['STP'] > 900, 'STP'] = np.nan
    # data = data.dropna(subset=['STP'])


    # пункт 1
    # гистограммы
    fig, ax = plt.subplots()
    plt.hist2d(data['TEMP'], data['WDSP'], density=True, bins=20, cmap=cm.cividis)
    ax.set_xlabel('температура')
    ax.set_ylabel('скорость ветра')
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    plt.hist2d(data['TEMP'], data['SLP'], density=True, bins=20, cmap=cm.cividis)
    ax.set_xlabel('температура')
    ax.set_ylabel('давление')
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    plt.hist2d(data['SLP'], data['WDSP'], density=True, bins=20, cmap=cm.cividis)
    ax.set_xlabel('давление')
    ax.set_ylabel('скорость ветра')
    plt.colorbar()
    plt.show()

    # ядерная оценка
    density = stats.gaussian_kde((data['TEMP'], data['WDSP'], data['SLP']))

    temp_grid = np.linspace(min(data['TEMP']), max(data['TEMP']), 20)
    wind_grid = np.linspace(min(data['WDSP']), max(data['WDSP']), 20)
    pres_grid = np.linspace(min(data['SLP']), max(data['SLP']), 20)

    density_grid_tw = []
    for i, wind in enumerate(wind_grid):
        density_grid_tw.append([density((j, wind, data['SLP'].mean()))[0] for j in temp_grid])

    density_grid_tw = np.array(density_grid_tw)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    temp_grid_mesh, wind_grid_mesh = np.meshgrid(temp_grid, wind_grid)
    surf = ax.plot_surface(wind_grid_mesh, temp_grid_mesh, density_grid_tw, cmap=cm.cividis)
    fig.colorbar(surf)
    ax.set_ylabel('температура')
    ax.set_xlabel('скорость ветра')
    plt.show()

    density_grid_pw = []
    for i, wind in enumerate(wind_grid):
        density_grid_pw.append([density((data['TEMP'].mean(), wind, j))[0] for j in pres_grid])

    density_grid_pw = np.array(density_grid_pw)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    pres_grid_mesh, wind_grid_mesh = np.meshgrid(pres_grid, wind_grid)
    surf = ax.plot_surface(wind_grid_mesh, pres_grid_mesh, density_grid_pw, cmap=cm.cividis)
    fig.colorbar(surf)
    ax.set_ylabel('давление')
    ax.set_xlabel('скорость ветра')
    plt.show()

    # pd.plotting.scatter_matrix(data, diagonal="kde")
    # plt.tight_layout()
    # plt.show()
    # нам это не нужно, нo выглядит красиво)
    # sns.lmplot("TEMP", "STP", data, hue="SLP", fit_reg=False)
    # plt.show()

    # пункт 2
    mrv = data[['WDSP', 'TEMP', 'SLP']]
    # Мат ожидание и дисперсия
    print("mean")
    print(mrv.apply(np.mean))
    print("std")
    print(mrv.apply(np.std))
    print("var")
    print(mrv.apply(np.var))

    # пункт 3

    means = data[['WDSP', 'TEMP', 'SLP']].groupby('SLP').mean()
    # print("conditional mean\n", means)

    plt.scatter(means.index, means['WDSP'])
    plt.title("мат ожидание скорости ветра при фиксированных значениях давления")
    plt.show()

    plt.scatter(means.index, means['TEMP'])
    plt.title("мат ожидание температуры при фиксированных значениях давления")
    plt.show()

    # Условная дисперсия


    # пункт 4
    corr = data.corr()
    print(corr)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
    plt.show()

    # пункт 6
    X = data[['WDSP', 'TEMP']]
    y = data[['SLP']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    poly = PolynomialFeatures(2)
    X_train = poly.fit_transform(X_train)
    poly = PolynomialFeatures(2)
    X_test = poly.fit_transform(X_test)
    X = poly.fit_transform(X)

    reg = LinearRegression(normalize=True)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_all = np.array(reg.predict(X))

    x = range(len(y_test))
    plt.scatter(x, y_test, label=u'Реальное значение')
    plt.scatter(x, y_pred, label=u'Предсказанное линейной моделью')
    plt.legend()
    plt.show()

    # Вычисление метрик модели
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean absolute error = ', mae)
    print('Mean squared error = ', mse)

    # пункт 7
    # Построение распределения остатков
    y1 = np.array(y)
    y2 = np.array(y_pred_all)
    y_diff = y1[:, 0] - y2[:, 0]
    # plt.hist(y_diff, 30, alpha=.3, density=True)
    sns.distplot(y_diff, kde=False)
    plt.show()

    # Confidence interval of regression coef
    mod = sm.OLS(y_train, X_train)
    res = mod.fit()
    print(res.conf_int(0.01))