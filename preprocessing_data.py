import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.stats import kde


def main():
    # Cчитываем файл, берем только станцию, дату, температуру, скорость ветра, индекс - дата
    tb_3cols = pd.read_csv("data/data_one_dim.csv", index_col=1, na_values='NA',
                           usecols=['STATION', 'DATE', 'TEMP', 'WDSP'])
    # кол-во уникальных станций
    stations = tb_3cols['STATION'].unique().tolist()
    print(len(stations))

    # заполнение nan
    # average = tb_3cols.loc[m1:m2]
    # tb_3cols.fillna(value={'TEMP': average['TEMP'].mean()}, inplace=True)

    # ax = tb_3cols.plot(figsize=(16, 5), title='Temperature')
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Temperature")
    # plt.show()

    # Создаем датасет со средним значением для каждого месяца
    cols = list(['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    all_files = []
    for station in stations:
        cols_values = list()
        month = tb_3cols.loc[tb_3cols['STATION'] == station]
        for y in range(2000, 2020):
            cols_year_values = []
            for m in range(1, 13):
                # берем одну стацию, один месяц
                if m < 10:
                    m1 = '{0}-0{1}-01'.format(y, m)
                    m2 = '{0}-0{1}-31'.format(y, m)
                    m3 = '{0}-0{1}-28'.format(y, m)
                else:
                    m1 = '{0}-{1}-01'.format(y, m)
                    m2 = '{0}-{1}-31'.format(y, m)
                    m3 = '{0}-{1}-28'.format(y, m)
                tr = np.mean(month['TEMP'].loc[m1:m2])
                if pd.isna(tr):
                    average = tb_3cols['TEMP'].loc[m1:m3]
                    cols_year_values.append((np.mean(average.tolist()) - 32) / 1.8)
                else:
                    month_temp = month.loc[m1:m2]
                    cols_year_values.append((np.mean(month_temp['TEMP'].tolist()) - 32) / 1.8)
            cols_values.append([y] + cols_year_values)

        table_avg_temp = pd.DataFrame(cols_values, columns=cols)

        print(table_avg_temp)
        table_avg_temp.to_csv(f'data/average_temp_{station}.csv')
        all_files.append(f'data/average_temp_{station}.csv')

    #optional join stations in one dataset
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv("data/average_temp.csv")


if __name__ == '__main__':
    main()
