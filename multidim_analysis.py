import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm

if __name__ == "__main__":
    data = pd.read_csv("data/data_spb.csv",
                       usecols=['STATION', 'DATE', 'TEMP', 'SLP', 'WDSP', 'STP'], index_col=0)
    data = data.loc[26063099999]
    data = data.replace(9999.9, np.nan, regex=True)
    # TEMP - Mean temperature (.1 Fahrenheit)
    # DEWP - Mean dew point (.1 Fahrenheit)
    # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9 (.1 mb)
    # VISIB - Mean visibility (.1 miles)
    # WDSP – Mean wind speed (.1 knots)
    # MXSPD - Maximum sustained wind speed (.1 knots)
    # GUST - Maximum wind gust (.1 knots)
    # MAX - Maximum temperature (.1 Fahrenheit)
    # MIN - Minimum temperature (.1 Fahrenheit)
    # PRCP - Precipitation amount (.01 inches)
    # SNDP - Snow depth (.1 inches)
    print(data.head())
    print(data['SLP'])

    # пункт 4
    corr = data.corr()
    print(corr)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
    plt.show()

    # пункт 6
    data = data.dropna(subset=['SLP'])
    X = data[['WDSP', 'TEMP', 'STP']]
    y = data[['SLP']]

    # нормировка
    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X))
    # X.columns = ['SLP', 'VISIB', 'FRSHTT', 'LONGITUDE']
    # scaler = StandardScaler()
    # y = pd.DataFrame(scaler.fit_transform(y))
    # y.columns = ['TEMP']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 2 степень. закомментить - будет линейная
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
    sns.distplot(y_diff, kde=False)
    plt.show()

    # Confidence interval of regression coef
    mod = sm.OLS(y_train, X_train)
    res = mod.fit()
    print(res.conf_int(0.01))