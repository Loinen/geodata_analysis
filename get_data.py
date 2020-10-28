"""
The module downloads data from the www.ncei.noaa.gov website
The file data/stations.csv is from https://gis.ncdc.noaa.gov/maps/ncei/cdo/daily
"""
import os
import wget
import pandas as pd


if __name__ == '__main__':
    table = pd.read_csv("data/stations.csv", index_col=1,
                           usecols=['STATION', 'STATION_ID'])
    stations = table['STATION_ID'].unique().tolist()
    print(len(stations))

    stations = [stations[i:i + 50] for i in range(0, len(stations), 50)]

    all_files = []

    for i, stations_batch in enumerate(stations):
        stations_str = ','.join(map(str, stations_batch))

        url = f"https://www.ncei.noaa.gov/access/services/data/v1" \
              f"?dataset=global-summary-of-the-day&dataTypes" \
              f"=LATITUDE,LONGITUDE,ELEVATION,NAME,TEMP,TEMP_ATTRIBUTES," \
              f"DEWP,DEWP_ATTRIBUTES,SLP,SLP_ATTRIBUTES,STP,STP_ATTRIBUTES," \
              f"VISIB,VISIB_ATTRIBUTES,WDSP,WDSP_ATTRIBUTES,MXSPD,GUST,MAX," \
              f"MAX_ATTRIBUTES,MIN,MIN_ATTRIBUTES,PRCP,PRCP_ATTRIBUTES,SNDP," \
              f"FRSHTT&stations={stations_str}&startDate=2010-01-01&endDate=2020-01-02"

        filename = f"data/data_{i}.csv"
        if os.path.exists(filename):
            os.remove(filename)
        wget.download(url, filename)

        all_files.append(filename)

    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv("data/all_stations_data.csv")

    for file_name in all_files:
        os.remove(file_name)
