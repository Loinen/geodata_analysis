import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def csv_reader(file_obj):
    # Открываем файл
    reader = csv.reader(file_obj)
    for row in reader:
        print(" ".join(row))


# csv_path = "global-summary-of-the-day-2020-10-23T07-37-38.csv"
# with open(csv_path, "r") as f_obj:
#     csv_reader(f_obj)

# Cчитываем файл
# tb_data = pd.read_csv("global-summary-of-the-day-2020-10-23T07-37-38.csv", index_col=0)
# print(tb_data)
# Или берем только станцию, дату, температуру, индекс - дата
tb_3cols = pd.read_csv("global-summary-of-the-day-2020-10-23T07-37-38.csv", index_col=1,
                      usecols=['STATION', 'DATE', 'TEMP'])
print(tb_3cols)
# кол-во уникальных станций
print(len(tb_3cols['STATION'].unique()))
stations = tb_3cols['STATION'].unique().tolist()
print(stations)
# Берем строку с выбранной датой
print("2010",stations[0])
# print(tb_3cols.loc[['2010-01-01']])
# print("2011")
dates = pd.date_range(start='2010-01-01',end='2010-02-01', closed='left',freq='D')
# print(dates.tolist())

# print(tb_3cols.loc[['2010-01-01'],[tb_3cols['STATION'] == stations[0]]])
#query = np.where((tb_3cols['STATION'] == stations[0]) & (tb_3cols.str.startswith('2')))

# print(tb_3cols.loc[tb_3cols.str.startswith('2')])

# Создаем датасет со средним значением для каждого месяца
# сейчас тут скорее псевдокод
for station in stations:
    cols_values = list()
    for y in range(2010, 2020):
        for m in range(1, 12):
        # берем одну стацию, один месяц
            month = tb_3cols.loc[tb_3cols['STATION'] == station]
            month = month.loc['{0}-{1}-01':'{0}-{1}-31'.format(y, m)]
            print(month)
            cols_values.append(np.mean(month['TEMP'].tolist()))
            print(month['TEMP'].tolist(), cols_values)
            # столбцы - Jan, Feb, Mer, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
            # так создаем новую таблицу (это пример с документации)
            dfl = pd.DataFrame(np.random.randn(5, 4), columns = list('ABCD'),
                               index=pd.date_range('20130101',periods=5))
            # Тут добавляем строку в датафрейм (тоже пример)
            df = pd.DataFrame([[1, 'Bob', 8000],
                               [2, 'Sally', 9000],
                               [3, 'Scott', 20]], columns=['id', 'name', 'power level'])
            df.append(df.sum(axis=0), ignore_index=True)
    # df_avg_temp1 = pd.DataFrame
    # df_avg_temp = pd.concat([df_avg_temp1, df_avg_temp2], ignore_index=True)
    # df_avg_temp.merge(df_avg_temp, left_on='DATE', right_on='DATE', suffixes=('_left', '_right'))

month.to_csv('result.csv')

# tb_4cols = pd.read_csv("global-summary-of-the-day-2020-10-23T07-37-38.csv", index_col=1,
#                       usecols=['STATION', 'DATE', 'TEMP', 'WDSP'])
# print(tb_4cols)

