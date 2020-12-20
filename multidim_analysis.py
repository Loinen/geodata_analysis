import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from math import sqrt


if __name__ == "__main__":
    data = pd.read_csv("data/data_spb.csv",
                       usecols=['STATION', 'DATE', 'TEMP', 'SLP', 'WDSP'], index_col=1)
    # тут параметры с норм корреляцией(0,5) хотя бы с 1
    #                    usecols=['STATION', 'DATE', 'GUST', 'SLP',
    #                             'SNDP', 'STP', 'TEMP',  'WDSP'], index_col=0)
    data = data.loc[data.STATION == 26063099999]
    data = data.drop(columns='STATION')

    # TEMP - Mean temperature (.1 Fahrenheit)
    # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9 (.1 mb)
    # WDSP – Mean wind speed (.1 knots)

    # удаление пропущенных значений
    missing_vals = data.loc[data.SLP==9999.9]
    data = data.replace(9999.9, np.nan, regex=True)
    data = data.dropna(subset=['SLP'])

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

    density_grid_pt = []
    for i, temp in enumerate(temp_grid):
        density_grid_pt.append([density((temp, data['WDSP'].mean(), j))[0] for j in pres_grid])

    density_grid_pt = np.array(density_grid_pt)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    pres_grid_mesh, temp_grid_mesh = np.meshgrid(pres_grid, temp_grid)
    surf = ax.plot_surface(temp_grid_mesh, pres_grid_mesh, density_grid_pt, cmap=cm.cividis)
    fig.colorbar(surf)
    ax.set_ylabel('давление')
    ax.set_xlabel('температура')
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

    means = data[['WDSP', 'TEMP', 'SLP']].groupby(['WDSP', 'TEMP']).mean()
    wind = []
    temp = []
    for ind in means.index:
        wind.append(ind[0])
        temp.append(ind[1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(wind, temp, means.SLP, cmap='viridis')
    ax.set_ylabel('температура')
    ax.set_xlabel('скорость ветра')
    plt.show()

    # Условная дисперсия
    cond_var = data[['WDSP', 'TEMP', 'SLP']].groupby(['WDSP', 'TEMP']).var()
    cond_var = cond_var.dropna(subset=['SLP'])

    wind = []
    temp = []
    for ind in cond_var.index:
        wind.append(ind[0])
        temp.append(ind[1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(wind, temp, cond_var.SLP, cmap='viridis')
    ax.set_ylabel('температура')
    ax.set_xlabel('скорость ветра')
    plt.show()

    # пункт 4
    corr = data.corr()
    print(corr)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm')
    plt.show()

    # пункт 6

    # multiple correlation
    mult_corr = sqrt(corr.SLP[2] ** 2 + corr.SLP[1] ** 2 - \
                     2 * corr.TEMP[2] * corr.SLP[2] * corr.SLP[1]) \
                     / (1 - corr.TEMP[2] ** 2)
    print(f"Множественный коэффициент корреляции давления на ветер и темппературу = {mult_corr}")


    X = data[['WDSP', 'TEMP']]
    y = data[['SLP']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # poly = PolynomialFeatures(2)
    # X_train = poly.fit_transform(X_train)
    # poly = PolynomialFeatures(2)
    # X_test = poly.fit_transform(X_test)
    # X = poly.fit_transform(X)

    reg = LinearRegression(normalize=True)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_all = np.array(reg.predict(X))

    x = range(len(y_test))
    plt.scatter(x, y_test, color='navy', label=u'Реальное значение')
    plt.scatter(x, y_pred, color='gold', label=u'Предсказанное линейной моделью')
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
    sns.distplot(y_diff, kde=False)
    plt.show()

    print(f"pvalue, что распределение остатков гауссовское = {stats.normaltest(y_diff).pvalue}")

    # коэффициент детерминации
    determ = 1 - sum(y_diff**2)/np.var(y1[:, 0]) / len(y_diff)
    print(f"Коэффициент детерминации = {determ}")

    # заполнение пропусков
    X_pred = missing_vals[['WDSP', 'TEMP']]
    y_pred = reg.predict(X_pred)

    missing_vals['SLP'] = y_pred

    data['SLP'].plot(color='navy', label='Реальные значения')
    missing_vals['SLP'].plot(color='gold', label='Заполненные значения')

    plt.legend()
    plt.show()
