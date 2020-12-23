import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from copy import copy
import numpy as np
import networkx as nx

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from pgmpy.base import DAG


def draw_comparative_hist(parametr: str, original_data: pd.DataFrame, data_sampled: pd.DataFrame):
    final_df = pd.DataFrame()
    df1 = pd.DataFrame()
    df1[parametr] = original_data[parametr]
    df1['Data'] = 'Original data'
    df1['Probability'] = df1[parametr].apply(
        lambda x: (df1.groupby(parametr)[parametr].count()[x]) / original_data.shape[0])
    df2 = pd.DataFrame()
    df2[parametr] = data_sampled[parametr]
    df2['Data'] = 'Synthetic data'
    df2['Probability'] = df2[parametr].apply(
        lambda x: (df2.groupby(parametr)[parametr].count()[x]) / data_sampled.shape[0])
    final_df = pd.concat([df1, df2])
    sns.barplot(x=parametr, y="Probability", hue="Data", data=final_df)
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


def sampling(bn: DAG, data: pd.DataFrame, n: int = 100):
    bn_new = BayesianModel(bn.edges())
    bn_new.fit(data)
    sampler = BayesianModelSampling(bn_new)
    sample = sampler.forward_sample(size=n, return_type='dataframe')
    return sample


if __name__ == "__main__":
    # data = pd.read_csv("data/data_spb.csv",
    #                    usecols=['STATION', 'TEMP', 'WDSP', 'SLP', 'GUST', 'SNDP'], index_col=0)

    data = pd.read_csv("data/data_spb.csv", usecols=['STATION', 'SLP', 'DEWP', 'MAX',
                                                     'MIN', 'TEMP', 'WDSP'], index_col=0)
    data = data.loc[26063099999]
    # TEMP - Mean temperature (.1 Fahrenheit)
    # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9 (.1 mb)
    # WDSP – Mean wind speed (.1 knots)

    # удаление пропущенных значений
    data = data.replace(9999.9, np.nan, regex=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)

    # Корреляционная матрица
    corr = data.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
    plt.show()

    pd.options.display.max_rows = 500
    pd.options.display.max_columns = 500
    print(data.head(201))
    print(len(data))
    # ограничиваем число данных, выбираем параметры для ручного построения байесовской сети
    data = data[1:2000]
    data2 = data[['DEWP', 'SLP', 'TEMP', 'WDSP']]

    bins = 4
    # на случай, если бинов недостаточно
    # while True:
    #     try:
    #
    #         break
    #
    #     except KeyError:
    #         print(bins)
    #         bins = bins+1
    #         continue
    transformed_data = copy(data)
    transformed_data2 = copy(data2)

    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    est2 = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')

    data_discrete = est.fit_transform(transformed_data.values[:, 0:6])
    transformed_data[['DEWP', 'MAX', 'MIN', 'SLP', 'TEMP', 'WDSP']] = data_discrete
    hc = HillClimbSearch(transformed_data, scoring_method=BDeuScore(transformed_data))
    best_model = hc.estimate()

    G_K2 = nx.DiGraph()
    G_K2.add_edges_from(best_model.edges())
    pos = nx.layout.circular_layout(G_K2)
    nx.draw(G_K2, pos, with_labels=True, font_weight='bold')
    plt.show()

    accuracy = accuracy_params_restoration(BayesianModel(best_model.edges()), transformed_data)
    print(accuracy)

    # ручное
    data_discrete2 = est2.fit_transform(data2.values[:, 0:4])
    transformed_data2[['DEWP', 'SLP', 'TEMP', 'WDSP']] = data_discrete2
    hc = HillClimbSearch(transformed_data2, scoring_method=BDeuScore(transformed_data2))
    best_model2 = hc.estimate()

    # создаем узлы для байесовской сети
    bm = BayesianModel([('TEMP', 'SLP'), ('WDSP', 'SLP'), ('DEWP', 'TEMP')])

    G_K2 = nx.DiGraph()
    G_K2.add_edges_from(bm.edges())
    pos = nx.layout.circular_layout(G_K2)
    nx.draw(G_K2, pos, with_labels=True, font_weight='bold')
    plt.show()
    print(bm.get_cpds())
    print(transformed_data2)

    accuracy2 = accuracy_params_restoration(BayesianModel(best_model2.edges()), transformed_data2)
    ##
    hc_BicScore = HillClimbSearch(transformed_data, scoring_method=BicScore(transformed_data))
    best_model_BicScore = hc_BicScore.estimate()

    print("shape", transformed_data.shape)

    sample_Bic = sampling(best_model_BicScore, transformed_data, len(data))
    sample_Bic[['DEWP', 'MAX', 'MIN', 'SLP',  'TEMP', 'WDSP']] = est.inverse_transform(sample_Bic[
               ['DEWP', 'MAX', 'MIN', 'SLP',  'TEMP', 'WDSP']].values)

    draw_comparative_hist('DEWP', transformed_data, sample_Bic)
    draw_comparative_hist('SLP', transformed_data, sample_Bic)
    draw_comparative_hist('TEMP', transformed_data, sample_Bic)
    draw_comparative_hist('WDSP', transformed_data, sample_Bic)

    hc_BicScore = HillClimbSearch(transformed_data2, scoring_method=BicScore(transformed_data2))
    best_model_BicScore = hc_BicScore.estimate()
    sample_Bic2 = sampling(best_model_BicScore, transformed_data2, len(data2))
    sample_Bic2[['DEWP', 'SLP', 'TEMP', 'WDSP']] = est2.inverse_transform(sample_Bic2[
                ['DEWP', 'SLP', 'TEMP', 'WDSP']].values)

    draw_comparative_hist('DEWP', transformed_data2, sample_Bic2)
    draw_comparative_hist('SLP', transformed_data2, sample_Bic2)
    draw_comparative_hist('TEMP', transformed_data2, sample_Bic2)
    draw_comparative_hist('WDSP', transformed_data2, sample_Bic2)

    sns.distplot(data['WDSP'], label='Original data')
    sns.distplot(sample_Bic['WDSP'], label='Generated data dist')
    plt.legend()
    plt.show()

    sns.distplot(data['TEMP'], label='Original data')
    sns.distplot(sample_Bic['TEMP'], label='Generated data')
    plt.legend()
    plt.show()

    sns.distplot(data['SLP'], label='Original data')
    sns.distplot(sample_Bic['SLP'], label='Generated data')
    plt.legend()
    plt.show()

    sns.distplot(data['DEWP'], label='Original data')
    sns.distplot(sample_Bic['DEWP'], label='Generated data')
    plt.legend()
    plt.show()

    print(accuracy)
    print(accuracy2)
    print("bins", bins)
    print(sample_Bic['SLP'])
