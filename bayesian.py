import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from copy import copy
import numpy as np
import statsmodels.api as sm
import networkx as nx

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from pgmpy.base import DAG


if __name__ == "__main__":
    data = pd.read_csv("data/data_spb.csv",
                       usecols=['STATION', 'TEMP', 'SLP', 'WDSP', 'STP'], index_col=0)
    data = data.loc[26063099999]

    # TEMP - Mean temperature (.1 Fahrenheit)
    # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9 (.1 mb)
    # WDSP – Mean wind speed (.1 knots)

    # удаление пропущенных значений
    data = data.replace(9999.9, np.nan, regex=True)
    data = data.dropna(subset=['SLP'])
    data.loc[data['STP'] > 900, 'STP'] = np.nan
    data = data.dropna(subset=['STP'])
    data.reset_index(inplace=True, drop=True)

    # sns.displot(data['SLP'])
    # plt.show()
    # sns.displot(data['TEMP'])
    # plt.show()
    # sns.displot(data['WDSP'])
    # plt.show()
    # sns.displot(data['STP'])
    # plt.show()

    print(data.head(10))

    transformed_data = copy(data)
    est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
    data_discrete = est.fit_transform(data.values[:, 0:4])
    print(data_discrete)
    transformed_data[['SLP', 'STP', 'TEMP', 'WDSP']] = data_discrete

    hc = HillClimbSearch(transformed_data, scoring_method=K2Score(transformed_data))

    best_model = hc.estimate()

    G_K2 = nx.DiGraph()
    G_K2.add_edges_from(best_model.edges())
    pos = nx.layout.circular_layout(G_K2)
    nx.draw(G_K2, pos, with_labels=True, font_weight='bold')

    def accuracy_params_restoration(bn: BayesianModel, data: pd.DataFrame):
        bn.fit(data)
        result = pd.DataFrame(columns=['Parameter', 'accuracy'])
        bn_infer = VariableElimination(bn)
        for j, param in enumerate(data.columns):
            accuracy = 0
            test_param = data[param]
            test_data = data.drop(columns=param)
            evidence = test_data.to_dict('records')
            predicted_param = []
            for element in evidence:
                prediction = bn_infer.map_query(variables=[param], evidence=element)
                predicted_param.append(prediction[param])
            accuracy = accuracy_score(test_param.values, predicted_param)
            result.loc[j, 'Parameter'] = param
            result.loc[j, 'accuracy'] = accuracy
        return result

    accuracy_k2 = accuracy_params_restoration(BayesianModel(best_model.edges()), transformed_data)
    print(accuracy_k2)

    # # пункт 1
    # pd.plotting.scatter_matrix(data, diagonal="kde")
    # plt.tight_layout()
    # plt.show()
    # # нам это не нужно, нo выглядит красиво)
    # sns.lmplot("TEMP", "STP", data, hue="SLP", fit_reg=False)
    # plt.show()
    #
    # # пункт 2,3
    # X = data[['WDSP', 'TEMP', 'STP']]
    # y = data.SLP
    # # Мат ожидание
    # print("mean", X.apply(np.mean))
    # print("std", X.apply(np.std))
    # means = data[['WDSP', 'TEMP', 'STP', 'SLP']].groupby('SLP').mean()
    # print("conditional mean\n", means)
    #
    # # Дисперсия
    # X = [data['WDSP'], data['TEMP'], data['STP']]
    # sns.heatmap(np.cov(X), annot=True, fmt='g')
    # plt.show()
    # print(np.cov(X))
    #
    # # пункт 4
    # corr = data.corr()
    # print(corr)
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
    # plt.show()
    #
    # # пункт 6
    # X = data[['WDSP', 'TEMP', 'STP']]
    # y = data[['SLP']]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #
    # # 2 степень
    # poly = PolynomialFeatures(2)
    # X_train = poly.fit_transform(X_train)
    # poly = PolynomialFeatures(2)
    # X_test = poly.fit_transform(X_test)
    # X = poly.fit_transform(X)
    #
    # reg = LinearRegression(normalize=True)
    # reg.fit(X_train, y_train)
    # y_pred = reg.predict(X_test)
    # y_pred_all = np.array(reg.predict(X))
    #
    # x = range(len(y_test))
    # plt.scatter(x, y_test, label=u'Реальное значение')
    # plt.scatter(x, y_pred, label=u'Предсказанное линейной моделью')
    # plt.legend()
    # plt.show()
    #
    # # Вычисление метрик модели
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # print('Mean absolute error = ', mae)
    # print('Mean squared error = ', mse)
    #
    # # пункт 7
    # # Построение распределения остатков
    # y1 = np.array(y)
    # y2 = np.array(y_pred_all)
    # y_diff = y1[:, 0] - y2[:, 0]
    # sns.distplot(y_diff, kde=False)
    # plt.show()
    #
    # # Confidence interval of regression coef
    # mod = sm.OLS(y_train, X_train)
    # res = mod.fit()
    # print(res.conf_int(0.01))