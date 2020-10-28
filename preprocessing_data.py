import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.stats import kde


def main():
    # Cчитываем файл
    # tb_data = pd.read_csv("global-summary-of-the-day-2020-10-23T07-37-38.csv", index_col=0)
    # Или берем только станцию, дату, температуру, индекс - дата
    tb_3cols = pd.read_csv("data/data_1.csv", index_col=1,
                           usecols=['STATION', 'DATE', 'TEMP', 'WDSP'])
    # print(tb_3cols)
    # кол-во уникальных станций
    stations = tb_3cols['STATION'].unique().tolist()
    # print(stations)

    # tb_2cols = pd.read_csv("global-summary-of-the-day-2020-10-23T07-37-38.csv", index_col=0,
    #                        usecols=['DATE', 'TEMP'])
    # ax = tb_2cols.plot(figsize=(16, 5), title='Temperature')
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Temperature")
    # plt.show()

    # Создаем датасет со средним значением для каждого месяца
    cols = list(['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    for station in stations:
        cols_values = list()
        month = tb_3cols.loc[tb_3cols['STATION'] == station]
        for y in range(2010, 2020):
            cols_year_values = []
            for m in range(1, 13):
                # берем одну стацию, один месяц
                if m < 10:
                    m1 = '{0}-0{1}-01'.format(y, m)
                    month_temp = month.loc[m1:'{0}-0{1}-31'.format(y, m)]
                else:
                    m1 = '{0}-{1}-01'.format(y, m)
                    m2 = '{0}-{1}-31'.format(y, m)
                    month_temp = month.loc[m1:m2]
                cols_year_values.append((np.mean(month_temp['TEMP'].tolist()) - 32) / 1.8)
                print(cols_year_values)
            cols_values.append([y] + cols_year_values)

        # print(cols_values)

        table_avg_temp = pd.DataFrame(cols_values, columns=cols)

    table_avg_temp.to_csv('average_temp.csv')
    print(table_avg_temp)


if __name__ == '__main__':
    main()
