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
                       usecols=['STATION', 'TEMP', 'SLP', 'WDSP', 'GUST', 'SNDP'], index_col=0)
    data = data.loc[26063099999]

    # TEMP - Mean temperature (.1 Fahrenheit)
    # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9 (.1 mb)
    # WDSP – Mean wind speed (.1 knots)

    # удаление пропущенных значений
    data = data.replace(9999.9, np.nan, regex=True)
    data = data.dropna(subset=['SLP'])
    data.loc[data['GUST'] > 990, 'GUST'] = np.nan
    data.loc[data['SNDP'] > 990, 'SNDP'] = np.nan
    data = data.dropna(subset=['GUST'])
    data = data.dropna(subset=['SNDP'])
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
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
    data_discrete = est.fit_transform(data.values[:, 0:5])
    print(data_discrete)
    transformed_data[['GUST', 'SLP', 'SNDP', 'TEMP', 'WDSP']] = data_discrete

    hc = HillClimbSearch(transformed_data, scoring_method=K2Score(transformed_data))

    best_model = hc.estimate()

    G_K2 = nx.DiGraph()
    G_K2.add_edges_from(best_model.edges())
    pos = nx.layout.circular_layout(G_K2)
    nx.draw(G_K2, pos, with_labels=True, font_weight='bold')
    plt.show()

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